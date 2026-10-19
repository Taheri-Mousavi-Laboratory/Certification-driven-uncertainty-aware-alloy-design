#!/usr/bin/env python3
"""
Alloy-718 yield-strength framework
==============================

Estimates the room-temperature yield strength of Alloy-718-type alloys by summing
four strengthening contributions computed from CALPHAD-style phase data:

    sigma_y = sigma_Orowan(gamma')            # Orowan bypass of gamma' precipitates
            + sigma_coherency(gamma'')        # coherency strengthening, oblate gamma''
            + sigma_ss                        # solid-solution strengthening (precomputed)
            + sigma_HallPetch(d)              # grain-boundary strengthening

The Hall-Petch term uses a single fixed grain size d (set with --grain-size or
in MaterialParams), applied to every composition.

DATA
----
No real data ships with this repository. By default the script generates
SYNTHETIC placeholder data so the workflow can be run end to end. The numbers
it produces are meaningless; they only demonstrate the method.

To use your own data, supply:

1. Phase data (HDF5) via --phase-data, containing these datasets
   (one row per composition, N rows total):

     MassFraction    (N, 7)  mass fractions in the order Ni, Cr, Nb, Mo, Ti, Al, Co
                             (Fe = balance)
     GP (mol %)      (N,)    gamma'  phase amount (mole fraction)
     GDP (mol %)     (N,)    gamma'' phase amount (mole fraction)
     Matrix (mol %)  (N,)    gamma matrix phase amount (mole fraction)
     LP_GP           (N,)    gamma'  lattice parameter (Angstrom)
     LP_GDP          (N,)    gamma'' lattice parameter (Angstrom)
     LP_M            (N,)    matrix lattice parameter (Angstrom)
     sigma_ss        (N,)    solid-solution strength (MPa)

2. A grain size in um via --grain-size (default in MaterialParams).

USAGE
-----
    python Alloy-718_yield_strength.py                       # synthetic demo
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants and model parameters
# ---------------------------------------------------------------------------

NA = 6.022e23  # Avogadro's number (1/mol)

# Element order inside the HDF5 "MassFraction" array
HDF5_ELEMENT_ORDER = ["Ni", "Cr", "Nb", "Mo", "Ti", "Al", "Co"]
# Column order used in the output table
OUTPUT_ELEMENT_ORDER = ["Ni", "Cr", "Nb", "Al", "Ti", "Co", "Mo"]


@dataclass
class MaterialParams:
    """Matrix properties and microstructural assumptions.

    All values are user-adjustable. Replace them with values appropriate to
    your alloy, processing route and test temperature.
    """
    G: float = 75.2e9          # matrix shear modulus (Pa)
    b: float = 0.254           # matrix Burgers vector (nm)
    nu: float = 0.33           # Poisson ratio of matrix
    M: float = 2.6             # Taylor factor

    # gamma' (spherical) precipitate size used in the Orowan term (nm)
    r_gp: float = 32.0

    # gamma'' (oblate) precipitate dimensions (nm)
    a_gdp: float = 200.0       # major dimension
    c_gdp: float = 33.0        # minor dimension

    # ln(R/r0) factor in the dislocation line tension
    line_tension_log_term: float = 55.0

    # Hall-Petch parameters
    sigma_0: float = 21.8      # intrinsic / friction stress (MPa)
    k_hp: float = 700.0        # Hall-Petch coefficient (MPa um^0.5)
    grain_size: float = 50.0   # fixed grain size used for every composition (um)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phase_data(path: Path) -> Dict[str, np.ndarray]:
    """Load phase data from an HDF5 file with the schema described above."""
    import h5py

    keys = {
        "conc": "MassFraction",
        "gp": "GP (mol %)",
        "gdp": "GDP (mol %)",
        "mat": "Matrix (mol %)",
        "LP_GP": "LP_GP",
        "LP_GDP": "LP_GDP",
        "LP_M": "LP_M",
        "sigma_ss": "sigma_ss",
    }
    with h5py.File(path, "r") as f:
        missing = [v for v in keys.values() if v not in f]
        if missing:
            raise KeyError(f"HDF5 file is missing datasets: {missing}")
        return {k: np.array(f[v]).astype(np.float64) for k, v in keys.items()}


def make_synthetic_phase_data(n: int = 200, seed: int = 0) -> Dict[str, np.ndarray]:
    """Generate random placeholder phase data with the same structure as the
    real input. Values are illustrative only and carry no physical meaning."""
    rng = np.random.default_rng(seed)

    # Mass fractions, order: Ni, Cr, Nb, Mo, Ti, Al, Co
    lo = np.array([0.50, 0.17, 0.045, 0.028, 0.006, 0.002, 0.000])
    hi = np.array([0.55, 0.21, 0.055, 0.033, 0.012, 0.008, 0.010])
    conc = rng.uniform(lo, hi, size=(n, 7))

    gp = rng.uniform(0.02, 0.08, n)
    gdp = rng.uniform(0.05, 0.15, n)
    mat = 1.0 - gp - gdp

    return {
        "conc": conc,
        "gp": gp,
        "gdp": gdp,
        "mat": mat,
        "LP_GP": rng.uniform(3.58, 3.62, n),
        "LP_GDP": rng.uniform(3.62, 3.66, n),
        "LP_M": rng.uniform(3.58, 3.61, n),
        "sigma_ss": rng.uniform(100, 300, n),
    }


# ---------------------------------------------------------------------------
# Hall-Petch
# ---------------------------------------------------------------------------

def hall_petch_strength(p: MaterialParams) -> float:
    """Hall-Petch strength (MPa) for the fixed grain size p.grain_size (um)."""
    if p.grain_size <= 0:
        raise ValueError("grain_size must be positive")
    return p.sigma_0 + p.k_hp / math.sqrt(p.grain_size)


# ---------------------------------------------------------------------------
# Precipitate strengthening
# ---------------------------------------------------------------------------

def molar_volume_cm3(lattice_param_A: float, phase_amount: float) -> float:
    """Volume (cm^3) of a phase from an FCC-type lattice parameter (Angstrom),
    assuming 4 atoms per unit cell."""
    return NA * 0.25 * lattice_param_A**3 * 1e-30 * phase_amount * 1e6


def volume_fraction(lp_p: float, amt_p: float, lp_m: float, amt_m: float) -> float:
    vp = molar_volume_cm3(lp_p, amt_p)
    vm = molar_volume_cm3(lp_m, amt_m)
    return vp / (vp + vm)


def orowan_strength(f: float, p: MaterialParams) -> float:
    """Orowan bypass stress (MPa) for spherical gamma' precipitates."""
    if f <= 0:
        return 0.0
    rbar = math.sqrt(2 / 3) * p.r_gp
    spacing = (math.sqrt(math.pi / (4 * f)) - 1) * 2 * rbar
    tau = (p.M * 0.4 * p.G * p.b * math.log(2 * rbar / p.b)
           / (math.pi * math.sqrt(1 - p.nu) * spacing))
    return tau / 1e6


def coherency_strength_oblate(
    f: float, lp_gdp: float, lp_m: float, p: MaterialParams
) -> float:
    """Coherency-strain strengthening (MPa) for strong, shearable oblate
    gamma'' precipitates (edge and screw dislocations averaged, times M)."""
    if f <= 0:
        return 0.0
    G, b, nu = p.G, p.b, p.nu

    r_eq = (p.a_gdp**2 * p.c_gdp) ** (1 / 3)
    rbar = math.sqrt(2 / 3) * r_eq
    spacing = (math.sqrt(math.pi / (4 * f)) - 1) * 2 * rbar

    # Effective lattice parameter of gamma'' and coherency strain
    c_gdp_lp = (lp_gdp**3 / 0.25) ** (1 / 3)
    strain = (1 / 3) * ((1 + nu) / (1 - nu)) * ((c_gdp_lp - lp_m) / lp_m)

    log_term = math.log(p.line_tension_log_term)
    pref = (G * b**2) / (4 * math.pi)
    T_edge = pref * ((1 + nu - 3 * nu * math.sin(math.pi / 2) ** 2) / (1 - nu)) * log_term
    T_screw = pref * ((1 + nu - 3 * nu * math.sin(0) ** 2) / (1 - nu)) * log_term

    def strong(T: float, coeff: float) -> float:
        num = T**3 * strain * G * rbar
        return coeff * (0.8 / spacing) * (num / b**3) ** 0.25

    tau_edge = strong(T_edge, 1.6327)
    tau_screw = strong(T_screw, 0.8435)

    return p.M * (tau_edge + tau_screw) / 2 / 1e6


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_yield_strength(
    phase: Dict[str, np.ndarray],
    p: Optional[MaterialParams] = None,
) -> pd.DataFrame:
    p = p or MaterialParams()
    conc = phase["conc"]

    # Compositions in wt %
    comp_wt = pd.DataFrame(conc * 100, columns=HDF5_ELEMENT_ORDER)[OUTPUT_ELEMENT_ORDER]

    orowan, coherency = [], []
    for j in range(len(conc)):
        f_gp = volume_fraction(phase["LP_GP"][j], phase["gp"][j],
                               phase["LP_M"][j], phase["mat"][j])
        f_gdp = volume_fraction(phase["LP_GDP"][j], phase["gdp"][j],
                                phase["LP_M"][j], phase["mat"][j])
        orowan.append(orowan_strength(f_gp, p))
        coherency.append(coherency_strength_oblate(
            f_gdp, phase["LP_GDP"][j], phase["LP_M"][j], p))

    sigma_hp = hall_petch_strength(p)
    orowan = np.array(orowan)
    coherency = np.array(coherency)
    sigma_ss = phase["sigma_ss"]

    out = comp_wt.add_suffix(" (wt%)")
    out["γ′ (mol %)"] = 100 * phase["gp"]
    out["γ″ (mol %)"] = 100 * phase["gdp"]
    out["Solid solution strength (MPa)"] = sigma_ss
    out["Orowan strength (MPa)"] = orowan
    out["Coherency strength (MPa)"] = coherency
    out["Grain size (µm)"] = p.grain_size
    out["Hall-Petch strength (MPa)"] = sigma_hp
    out["Yield strength (MPa)"] = np.sqrt(orowan**2 + coherency**2) + sigma_ss + sigma_hp
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--phase-data", type=Path, help="HDF5 file with phase data")
    ap.add_argument("--grain-size", type=float, default=MaterialParams.grain_size,
                    help="fixed grain size for Hall-Petch (um, default: %(default)s)")
    ap.add_argument("--output", type=Path, default=Path("yield_strength_results.csv"))
    ap.add_argument("--n-demo", type=int, default=200,
                    help="number of synthetic compositions when no data is given")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.phase_data:
        phase = load_phase_data(args.phase_data)
    else:
        print("No --phase-data given: using SYNTHETIC placeholder phase data.")
        phase = make_synthetic_phase_data(args.n_demo, args.seed)

    params = MaterialParams(grain_size=args.grain_size)
    results = compute_yield_strength(phase, params)

    results.to_csv(args.output, index=False)
    print(results.describe().T[["mean", "min", "max"]].round(2))
    print(f"\nSaved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()