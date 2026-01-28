"""
Strength estimation for Alloy 718

This script loads a Thermo-Calc/TC-Python-generated HDF5 dataset containing
phase fractions, lattice parameters. Then computes yield strength using:
  - Orowan strengthening (γ′)
  - Coherent strain strengthening (γ′′)
  - Solid-solution strengthening
  - Hall–Petch term (constant grain size assumption)

Outputs a pandas DataFrame with composition, phase fractions, and strength terms.

Notes:
- This is written as a standalone Python script.
- Constants and geometric assumptions are intentionally kept explicit for traceability.
"""


import math
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import os
import random
import math
import statistics

np.random.seed(42)

# ----------------------------
# Configuration / constants
# ----------------------------

# Mechanical constants (matrix)
G = 75.2e9          # Pa, matrix shear modulus
b = 0.254e-9        # m,  matrix Burgers vector (converted from nm)
nu = 0.33           # Poisson ratio
M = 2.6             # Taylor factor
NA = 6.022e23       # 1/mol, Avogadro number

# Hall–Petch parameters (grain size assumed constant here)
SIGMA_INTRINSIC = 21.8  # MPa
#K_LOCK = 158            # MPa * µm^1/2 (high-T, not used in final sum here)
K_CONSTANT = 700        # MPa * µm^1/2 (room temperature)
GRAIN_DIAMETER_UM = 50  # µm (placeholder / assumption)

# γ′ (Orowan) precipitate size assumption in nm
GAMMA_P_DIAMETER_NM = 32  # DOI: (https://www.tms.org/Superalloys/10.7449/1991/Superalloys_1991_337_350.pdf))


def load_hdf5_arrays(h5_path: Path) -> dict[str, np.ndarray]:
    """Load required arrays from the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        data = {
            "conc": np.array(f["MassFraction"], dtype=np.float32),
            "gp": np.array(f["GP (mol %)"], dtype=np.float32),
            "gdp": np.array(f["GDP (mol %)"], dtype=np.float32),
            "mat": np.array(f["Matrix (mol %)"], dtype=np.float32),
            "lp_gp": np.array(f["LP_GP"], dtype=np.float32),         # Lattice parameter of gamma prime
            "lp_gdp": np.array(f["LP_GDP"], dtype=np.float32),       # Lattice parameter of gamma double prime
            "lp_m": np.array(f["LP_M"], dtype=np.float32),           # Lattice parameter of the matrix (FCC phase)
            "sigma_ss": np.array(f["sigma_ss"], dtype=np.float32),   # Solid solution strengthening
        }
    return data


def composition_dataframe(conc: np.ndarray) -> pd.DataFrame:
    """
    Convert mass fractions into a composition DataFrame (wt%).
    Expected order in conc rows: [Ni, Cr, Nb, Mo, Ti, Al, Co], Fe is balance.
    """
    # wt% (vectorized)
    wt = conc.astype(np.float64) * 100.0
    df = pd.DataFrame(
        wt,
        columns=["Ni (wt%)", "Cr (wt%)", "Nb (wt%)", "Mo (wt%)", "Ti (wt%)", "Al (wt%)", "Co (wt%)"],
    )

    # Reorder to match your original output ordering (Mo at end)
    df = df[["Ni (wt%)", "Cr (wt%)", "Nb (wt%)", "Al (wt%)", "Ti (wt%)", "Co (wt%)", "Mo (wt%)"]]
    return df


def hall_petch_strength_rt() -> float:
    """Compute Hall–Petch strengthening (MPa) at room temperature assumption."""
    return SIGMA_INTRINSIC + (K_CONSTANT / math.sqrt(GRAIN_DIAMETER_UM))


def orowan_strengthening(
    gp_mol_percent: np.ndarray,
    mat_mol_percent: np.ndarray,
    lp_gp: np.ndarray,
    lp_m: np.ndarray,
) -> np.ndarray:
    """
    Orowan strengthening contribution from γ′ (MPa).

    Uses phase volume fraction estimated from mol% and lattice parameters.
    Assumes a fixed precipitate diameter for γ′.
    """
    # Convert diameter to meters
    r_nm = GAMMA_P_DIAMETER_NM / 2.0
    r_m = r_nm * 1e-9

    # Molar volumes (m^3/mol), based on fcc: V = 0.25 * a^3
    mol_vp = NA * 0.25 * (lp_gp.astype(np.float64) ** 3) * 1e-30
    mol_vm = NA * 0.25 * (lp_m.astype(np.float64) ** 3) * 1e-30

    # Volumes (arbitrary scale cancels in ratio); keep consistent factors
    vp = mol_vp * gp_mol_percent.astype(np.float64) * 1e6
    vm = mol_vm * mat_mol_percent.astype(np.float64) * 1e6

    volff = vp / (vp + vm)  # volume fraction (0–1)

    rbar = math.sqrt(2.0 / 3.0) * r_m
    lambdar = (np.sqrt(math.pi / (4.0 * volff)) - 1.0) * 2.0 * rbar

    # Orowan term (Pa)
    sigma_pa = (
        M
        * 0.4
        * G
        * b
        * np.log(2.0 * rbar / b)
        / (math.pi * math.sqrt(1.0 - nu) * lambdar)
    )
    return (sigma_pa / 1e6).astype(np.float64)  # MPa


def coherent_strain_strengthening_oblate(
    gdp_mol_percent: np.ndarray,
    mat_mol_percent: np.ndarray,
    lp_gdp: np.ndarray,
    lp_m: np.ndarray,
) -> np.ndarray:
    """
    Coherent strain strengthening for γ′′ treated as oblate precipitates (MPa).

    Geometric assumptions are preserved from the original script.
    """
    # Phase volume fraction estimate
    mol_vp_db = NA * 0.25 * (lp_gdp.astype(np.float64) ** 3) * 1e-30
    mol_vm = NA * 0.25 * (lp_m.astype(np.float64) ** 3) * 1e-30

    vp_db = mol_vp_db * gdp_mol_percent.astype(np.float64) * 1e6
    vm = mol_vm * mat_mol_percent.astype(np.float64) * 1e6
    volff = vp_db / (vp_db + vm)

    # Oblate geometry (nm)
    a_nm = 200.0
    c_nm = 33.0
    h = c_nm / a_nm

    r_ell_nm = ((a_nm**2) * c_nm) ** (1.0 / 3.0)
    rbar_m = math.sqrt(2.0 / 3.0) * (r_ell_nm * 1e-9)

    lambdar_ss = (np.sqrt(math.pi / (4.0 * volff)) - 1.0) * 2.0 * rbar_m
    
    # Strain along the c axis (using lattice parameters)
    strain_coh = (1.0 / 3.0) * ((1.0 + nu) / (1.0 - nu)) * ((lp_gdp - lp_m) / lp_m)

    # Line tension (Pa·m), keep original log(55) choice
    log_term = math.log(55.0)
    t_edge = (G * b**2 / (4.0 * math.pi)) * ((1.0 + nu) / (1.0 - nu)) * log_term
    t_screw = (G * b**2 / (4.0 * math.pi)) * ((1.0 + nu) / (1.0 - nu)) * log_term

    # Strong (shearable) oblate contributions
    coh_edge_1 = (t_edge**3) * np.abs(strain_coh) * G * rbar_m
    coh_screw_1 = (t_screw**3) * np.abs(strain_coh) * G * rbar_m

    coh_strong_edge = 1.6327 * (0.8 / lambdar_ss) * (coh_edge_1 / b**3) ** (1.0 / 4.0)
    coh_strong_screw = 0.8435 * (0.8 / lambdar_ss) * (coh_screw_1 / b**3) ** (1.0 / 4.0)

    sigma_pa = M * (coh_strong_edge + coh_strong_screw) / 2.0
    return (sigma_pa / 1e6).astype(np.float64)  # MPa


def build_strength_dataframe(h5_path: str | Path) -> pd.DataFrame:
    """Run the full strength estimation pipeline and return a DataFrame."""

    h5_path = Path(h5_path)
    data = load_hdf5_arrays(h5_path)

    conc = data["conc"]
    gp = data["gp"]
    gdp = data["gdp"]
    mat = data["mat"]
    lp_gp = data["lp_gp"]
    lp_gdp = data["lp_gdp"]
    lp_m = data["lp_m"]
    sigma_ss = data["sigma_ss"].astype(np.float64)

    comp_df = composition_dataframe(conc)

    sigma_orowan = orowan_strengthening(gp, mat, lp_gp, lp_m)
    sigma_coh = coherent_strain_strengthening_oblate(gdp, mat, lp_gdp, lp_m)
    sigma_hp_rt = hall_petch_strength_rt()

    yield_strength = sigma_orowan + sigma_coh + sigma_ss + sigma_hp_rt

    df = comp_df.copy()
    df["γ′(mole %)"] = (100.0 * gp.astype(np.float64))
    df["γ′′(mole %)"] = (100.0 * gdp.astype(np.float64))
    df["Solid solution strength (MPa)"] = sigma_ss
    df["Orowan strength (MPa)"] = sigma_orowan
    df["Coherent strain strength (MPa)"] = sigma_coh
    df["Hall–Petch strength (MPa)"] = sigma_hp_rt
    df["Yield strength (MPa)"] = yield_strength

    return df


def main() -> None:
    h5_file = Path("718_strength.hdf5")
    Alloy718_comp_to_strength_df = build_strength_dataframe(h5_file)
    print(Alloy718_comp_to_strength_df.head())

if __name__ == "__main__":
    main()
