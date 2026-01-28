"""
Local sensitivity analysis for Alloy 718 yield strength using a PCE surrogate.

This script uses ONLY the analytic gradient methodology (ChaosPy derivative of the PCE polynomial)
to compute local sensitivities dY/dx at a user-provided base composition, predicts the base yield
strength, then estimates the expected strength variation given user-specified wt.% perturbations.

Workflow:
  1) Load the trained PCE surrogate (pickled ChaosPy model)
  2) Ask user for a base composition (Ni, Cr, Nb, Al, Ti, Co, Mo) in wt.%
  3) Predict base yield strength (MPa)
  4) Compute analytic (PCE) gradient at the base composition
  5) Ask user for wt.% variations (ΔNi, ΔCr, ..., ΔMo) in the same order
  6) Compute per-element ΔY and the total expected ΔY
"""

from __future__ import annotations
import sys
import pickle
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import chaospy as cp


# ----------------------------
# Configuration
# ----------------------------
ELEMENTS: Tuple[str, ...] = ("Ni", "Cr", "Nb", "Al", "Ti", "Co", "Mo")


# ----------------------------
# Load surrogate model
# ----------------------------
def resource_path(relative_path: str) -> Path:
    """
    Resolve resource paths for both:
    - normal Python execution
    - PyInstaller one-file executables
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # PyInstaller extracts files to a temp directory
        base_path = Path(sys._MEIPASS)
    else:
        # Project root (one level above this script)
        base_path = Path(__file__).resolve().parent.parent

    return base_path / relative_path


SURROGATE_MODEL_PATH = resource_path("model/pce_surrogate.pkl")
print("Looking for surrogate at:", SURROGATE_MODEL_PATH)

if not SURROGATE_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Surrogate model file not found:\n  {SURROGATE_MODEL_PATH}\n"
        "Make sure the PCE model exists at model/pce_surrogate.pkl "
        "and is included in the PyInstaller build."
    )

with open(SURROGATE_MODEL_PATH, "rb") as f:
    surrogate_model = pickle.load(f)

def surrogate_predict(X: np.ndarray) -> np.ndarray:
    """
    Predict yield strength (MPa) from compositions X of shape (N, 7) in wt%.
    Expected order: [Ni, Cr, Nb, Al, Ti, Co, Mo]
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(ELEMENTS):
        raise ValueError(f"X must have shape (N, {len(ELEMENTS)}). Got {X.shape}.")
    return surrogate_model(*X.T)

# ----------------------------
# Analytic (PCE) derivative
# ----------------------------
def pce_analytic_gradient(
    base_comp_wt: Sequence[float],
    pce_poly_model,
    elements: Tuple[str, ...] = ELEMENTS,
) -> Dict[str, float]:
    """
    Compute local partial derivatives dY/dx at base_comp_wt using ChaosPy analytic derivatives.
    """
    x = np.asarray(base_comp_wt, dtype=float)
    if x.shape[0] != len(elements):
        raise ValueError(f"base_comp_wt must have length {len(elements)} (got {x.shape[0]}).")

    grads = [float(cp.derivative(pce_poly_model, j)(*x)) for j in range(len(elements))]
    return dict(zip(elements, grads))


# -------------------------------------------------------
# Convert gradients to expected strength variation ΔY
# -------------------------------------------------------
def expected_strength_variation_from_gradients(
    dYdx: Dict[str, float],
    delta_wt: Sequence[float],
    elements: Tuple[str, ...] = ELEMENTS,
    combine: str = "sum",
) -> Tuple[Dict[str, float], float]:
    """
    Convert local gradients dY/dx into expected strength variation given Δ(wt.%).
    """
    delta_wt = np.asarray(delta_wt, dtype=float)
    if delta_wt.shape[0] != len(elements):
        raise ValueError(f"delta_wt must have length {len(elements)} (got {delta_wt.shape[0]}).")

    per_element: Dict[str, float] = {}
    contribs = []

    for el, dwt in zip(elements, delta_wt):
        if el not in dYdx:
            raise KeyError(f"Gradient dict missing element '{el}'. Keys={list(dYdx.keys())}")
        dy = float(dYdx[el] * dwt)
        per_element[el] = dy
        contribs.append(dy)

    contribs = np.asarray(contribs, dtype=float)

    if combine == "sum":
        total = float(np.sum(contribs))
    elif combine == "abs":
        total = float(np.sum(np.abs(contribs)))
    elif combine == "rss":
        total = float(np.sqrt(np.sum(contribs**2)))
    else:
        raise ValueError("combine must be one of: 'sum', 'abs', 'rss'")

    return per_element, total


def prompt_base_composition(elements: Tuple[str, ...] = ELEMENTS) -> np.ndarray:
    """
    Prompt the user to enter a base composition in wt.% in the same order as `elements`.
    """
    vals = []
    print("\nEnter base composition in wt.% (in the following order):")
    print("  " + ", ".join(elements))
    for el in elements:
        v = float(input(f"{el} (wt.%): ").strip())
        vals.append(v)
    return np.asarray(vals, dtype=float)


def prompt_element_variations(
    elements: Tuple[str, ...] = ELEMENTS,
    absolute: bool = False,
) -> np.ndarray:
    """
    Prompt the user to enter wt.% variations for each element in order.

    If absolute=True, takes abs(value) to treat variations as magnitudes only.
    """
    vals = []
    print("\nEnter expected variations in wt.% (Δ for each element):")
    for el in elements:
        v = float(input(f"Δ{el} (wt.%): ").strip())
        vals.append(abs(v) if absolute else v)
    return np.asarray(vals, dtype=float)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("\n==============================================")
    print(" Alloy 718 – Local Sensitivity Analysis (PCE) ")
    print("==============================================\n")

    # 1) Ask for base composition
    theta = prompt_base_composition(elements=ELEMENTS)

    # 2) Predict base yield strength
    y0 = float(surrogate_predict(theta.reshape(1, -1))[0])
    print(f"\nPredicted base yield strength: {y0:.2f} MPa")

    # 3) Analytic (PCE) gradient at base composition
    dY_pce = pce_analytic_gradient(
        base_comp_wt=theta,
        pce_poly_model=surrogate_model,
        elements=ELEMENTS,
    )

    print("\nAnalytic sensitivities dY/dx (MPa per wt.%):")
    for el in ELEMENTS:
        print(f"  {el:>2s} : {dY_pce[el]: .6f}")

    # 4) Ask for element variations
    print("\nEnter expected composition variations (wt.%):")
    delta = prompt_element_variations(elements=ELEMENTS, absolute=False)

    # 5) Convert gradients to expected strength variation
    per_el, total = expected_strength_variation_from_gradients(
        dYdx=dY_pce,
        delta_wt=delta,
        elements=ELEMENTS,
        combine="sum",   # options: "sum", "abs", "rss"
    )

    print("\nExpected strength variation per element (MPa):")
    for el in ELEMENTS:
        print(f"  {el:>2s} : {per_el[el]: .3f}")

    print("\n----------------------------------------------")
    print(f"Total expected ΔYield Strength: {total: .3f} MPa")
    print("----------------------------------------------")


if __name__ == "__main__":
    main()
    input("\nPress ENTER to exit...")

