"""
Local sensitivity analysis for Alloy 718 yield strength using a PCE surrogate.

This script uses ONLY the finite-difference methodology (central difference with bound-aware steps)
to compute local sensitivities dY/dx at a base composition, then estimates the expected strength
variation given user-specified wt.% perturbations.

Workflow:
  1) Load the trained PCE surrogate (pickled ChaosPy model)
  2) Compute finite-difference gradient at a base composition
  3) Ask user for wt.% variations (ΔNi, ΔCr, ..., ΔMo) in the same order
  4) Compute per-element ΔY and the total expected ΔY
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, Sequence, Tuple, Optional

import numpy as np


# ----------------------------
# Configuration
# ----------------------------
ELEMENTS: Tuple[str, ...] = ("Ni", "Cr", "Nb", "Al", "Ti", "Co", "Mo")
SURROGATE_MODEL_PATH = Path("pce_surrogate.pkl")
RNG_SEED = 42
np.random.seed(RNG_SEED)


# ----------------------------
# Load surrogate model
# ----------------------------
if not SURROGATE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Surrogate model file not found: {SURROGATE_MODEL_PATH.resolve()}\n"
        "Make sure you trained and saved the PCE model first."
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


# --------------------------------
# Finite difference local gradient
# --------------------------------
def finite_difference_gradient(
    base_comp_wt: Sequence[float],
    model,
    step: float = 0.05,
    elements: Tuple[str, ...] = ELEMENTS,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Compute a bound-aware central-difference gradient at base_comp_wt:

        d_i ≈ [ f(x0 + h e_i) - f(x0 - h^- e_i) ] / (h + h^-)

    where:
      - h is reduced if x0 + h would exceed the upper bound
      - h^- is reduced if x0 - h would exceed the lower bound

    Parameters
    ----------
    base_comp_wt:
        Composition point (length 7) in wt.% and in the same order used during training.
    model:
        Callable surrogate. Must accept 7 scalars and return scalar prediction: model(*x).
    step:
        Nominal step size (wt.%). Actual forward/back steps are clipped by bounds.
    elements:
        Element names in the same order as base_comp_wt.
    lower_bounds, upper_bounds:
        Optional bounds for each dimension (length 7). If not provided, bounds are +/- inf.

    Returns
    -------
    grads:
        Dict mapping {element -> dY/d(element)} in MPa per wt.%.
    """
    x0 = np.asarray(base_comp_wt, dtype=float).copy()
    d = len(elements)
    if x0.shape[0] != d:
        raise ValueError(f"base_comp_wt must have length {d} (got {x0.shape[0]}).")

    lb = np.full(d, -np.inf) if lower_bounds is None else np.asarray(lower_bounds, dtype=float)
    ub = np.full(d,  np.inf) if upper_bounds is None else np.asarray(upper_bounds, dtype=float)
    if lb.shape[0] != d or ub.shape[0] != d:
        raise ValueError("lower_bounds and upper_bounds must have length 7 if provided.")

    grads = np.zeros(d, dtype=float)

    for j in range(d):
        # Bound-aware steps
        h_minus = min(float(step), max(0.0, x0[j] - lb[j]))
        h_plus = min(float(step), max(0.0, ub[j] - x0[j]))
        denom = h_plus + h_minus

        if denom <= 0.0:
            grads[j] = 0.0
            continue

        xm = x0.copy()
        xp = x0.copy()
        xm[j] = x0[j] - h_minus
        xp[j] = x0[j] + h_plus

        y_minus = float(model(*xm))
        y_plus = float(model(*xp))

        grads[j] = (y_plus - y_minus) / denom

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

    Parameters
    ----------
    dYdx:
        Dict {element -> dY/d(element)} in MPa per wt.%.
    delta_wt:
        Sequence of perturbations Δ in wt.% in the SAME order as `elements`.
        Use signed values to get signed ΔY, or absolute values for magnitude-only.
    combine:
        How to combine per-element contributions into a total:
          - "sum" : signed sum of ΔY_i (default)
          - "abs" : sum of |ΔY_i|
          - "rss" : sqrt(sum(ΔY_i^2)) (quadrature)

    Returns
    -------
    per_element_deltaY:
        Dict {element -> ΔY_element} in MPa.
    total_deltaY:
        Combined total variation in MPa.
    """
    delta_wt = np.asarray(delta_wt, dtype=float)
    if delta_wt.shape[0] != len(elements):
        raise ValueError(f"delta_wt must have length {len(elements)} (got {delta_wt.shape[0]}).")

    per_element = {}
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


def prompt_element_variations(
    elements: Tuple[str, ...] = ELEMENTS,
    absolute: bool = False,
) -> np.ndarray:
    """
    Prompt the user to enter wt.% variations for each element in order.

    Parameters
    ----------
    absolute:
        If True, uses absolute value (treat variations as magnitudes only).
        If False, keeps sign (recommended if you want signed ΔY).
    """
    vals = []
    for el in elements:
        v = float(input(f"Enter variation Δ{el} (wt.%): ").strip())
        vals.append(abs(v) if absolute else v)
    return np.asarray(vals, dtype=float)


# ----------------------------
# Example usage
# ----------------------------
def main() -> None:
    # Base composition (wt.%) in the same order: [Ni, Cr, Nb, Al, Ti, Co, Mo]
    theta = [53.0, 21.0, 5.0, 0.40, 1.00, 0.50, 3.20]

    # Optional bounds (recommended)
    lb = [50.0, 17.0, 4.75, 0.20, 0.65, 0.00, 2.80]
    ub = [55.0, 21.0, 5.55, 0.80, 1.15, 1.00, 3.30]

    # Finite-difference gradient
    dY_fd = finite_difference_gradient(
        base_comp_wt=theta,
        model=surrogate_model,
        step=0.01,
        lower_bounds=lb,
        upper_bounds=ub,
    )

    print("\nFinite-difference dY/dx (MPa per wt.%):")
    for el in ELEMENTS:
        print(f"  {el}: {dY_fd[el]: .6f}")

    # User-specified variations (must match ELEMENTS order)
    delta = prompt_element_variations(elements=ELEMENTS, absolute=False)

    # Convert gradients to expected strength variation
    per_el, total = expected_strength_variation_from_gradients(
        dYdx=dY_fd,
        delta_wt=delta,
        elements=ELEMENTS,
        combine="sum",   # options: "sum", "abs", "rss"
    )

    print("\nPer-element ΔY (MPa):")
    for el in ELEMENTS:
        print(f"  {el}: {per_el[el]: .3f}")

    print(f"\nTotal expected ΔY (MPa) [{ 'sum' }]: {total: .3f}")


if __name__ == "__main__":
    main()
