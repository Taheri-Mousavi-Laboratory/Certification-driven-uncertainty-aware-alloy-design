"""
local_sensitivity_analysis.py
-----------------------------

Compute local composition sensitivity of predicted Alloy 718 yield strength
using a trained Polynomial Chaos Expansion (PCE) surrogate model.

The local sensitivity is calculated from the analytical derivative of the PCE
surrogate with respect to each alloying element:

    dY/dx_i

where:
    Y   = predicted yield strength, MPa
    x_i = composition of element i, wt.%

For a user-defined composition perturbation, the corresponding first-order
strength change is estimated as:

    ΔY_i = (dY/dx_i) Δx_i

This script is designed for GitHub release. The trained PCE surrogate file is
not included in the repository. Users should train their own PCE model and
provide its local path.

Example
-------
python local_sensitivity_analysis.py \
    --model_path ../Model/PCE_surrogate_model.pkl \
    --base_composition "Ni:52,Cr:19,Nb:5,Al:0.5,Ti:1.0,Co:0.5,Mo:3.2" \
    --delta_composition "Ni:0.1,Cr:0.1,Nb:0.05,Al:0.05,Ti:-0.05,Co:0.05,Mo:0.1" \
    --output_dir ../Results/local_sensitivity

Required packages
-----------------
numpy
pandas
matplotlib
chaospy
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Sequence, Tuple

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. User settings
# =============================================================================

# Element order expected by the trained PCE model.
# Make sure this matches the feature order used during PCE training.
ELEMENTS: Tuple[str, ...] = ("Ni", "Cr", "Nb", "Al", "Ti", "Co", "Mo")

# Example default base composition in wt.%.
# This is only used if the user does not provide --base_composition.
DEFAULT_BASE_COMPOSITION = {
    "Ni": 52.0,
    "Cr": 19.0,
    "Nb": 5.0,
    "Al": 0.5,
    "Ti": 1.0,
    "Co": 0.5,
    "Mo": 3.2,
}

# Example default perturbation in wt.%.
# Positive and negative values are allowed.
# A negative value means the element is locally decreased.
DEFAULT_DELTA_COMPOSITION = {
    "Ni": 0.10,
    "Cr": 0.10,
    "Nb": 0.05,
    "Al": 0.05,
    "Ti": -0.05,
    "Co": 0.05,
    "Mo": 0.10,
}


# =============================================================================
# 2. Utility functions
# =============================================================================

def parse_composition_string(
    composition_string: str,
    elements: Tuple[str, ...] = ELEMENTS,
) -> Dict[str, float]:
    """
    Parse a composition string into a dictionary.

    Expected format:
        "Ni:52,Cr:19,Nb:5,Al:0.5,Ti:1.0,Co:0.5,Mo:3.2"

    Parameters
    ----------
    composition_string : str
        Composition string provided by the user.
    elements : tuple of str
        Allowed element names.

    Returns
    -------
    composition_dict : dict
        Dictionary mapping element names to numerical values.
    """

    composition_dict = {}

    if composition_string is None:
        return composition_dict

    pairs = composition_string.replace(" ", "").split(",")

    for pair in pairs:
        if not pair:
            continue

        if ":" not in pair:
            raise ValueError(
                f"Invalid composition entry: '{pair}'. "
                "Use the format Element:value, for example Ni:52."
            )

        element, value = pair.split(":", maxsplit=1)

        if element not in elements:
            raise ValueError(
                f"Element '{element}' is not recognized. "
                f"Allowed elements are: {elements}"
            )

        composition_dict[element] = float(value)

    return composition_dict


def composition_dict_to_array(
    composition_dict: Dict[str, float],
    elements: Tuple[str, ...] = ELEMENTS,
) -> np.ndarray:
    """
    Convert a composition dictionary to an array in the required element order.

    Parameters
    ----------
    composition_dict : dict
        Dictionary containing composition values.
    elements : tuple of str
        Element order required by the PCE model.

    Returns
    -------
    composition_array : ndarray, shape (n_elements,)
        Composition array ordered as ELEMENTS.
    """

    missing = [element for element in elements if element not in composition_dict]

    if missing:
        raise ValueError(
            "The following elements are missing from the composition input: "
            f"{missing}"
        )

    return np.array([composition_dict[element] for element in elements], dtype=float)


def load_pce_model(model_path: str):
    """
    Load a trained PCE surrogate model from a pickle file.

    The private model file is not included in the GitHub repository. Users should
    provide the local path to their own trained PCE model.

    Parameters
    ----------
    model_path : str
        Path to the pickled PCE model.

    Returns
    -------
    pce_model
        Loaded PCE surrogate model.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"PCE surrogate model was not found:\n{model_path.resolve()}\n\n"
            "Please provide the correct local path using --model_path.\n"
            "Example: --model_path ../Model/PCE_surrogate_model.pkl"
        )

    with open(model_path, "rb") as file:
        pce_model = pickle.load(file)

    return pce_model


def predict_with_pce(
    pce_model,
    composition_wt: Sequence[float],
    elements: Tuple[str, ...] = ELEMENTS,
) -> float:
    """
    Predict yield strength using the trained PCE surrogate.

    Parameters
    ----------
    pce_model
        Trained Chaospy PCE model.
    composition_wt : sequence of float
        Composition in wt.% in the order defined by ELEMENTS.
    elements : tuple of str
        Element order.

    Returns
    -------
    prediction : float
        Predicted yield strength in MPa.
    """

    x = np.asarray(composition_wt, dtype=float)

    if x.shape[0] != len(elements):
        raise ValueError(
            f"Expected composition length {len(elements)}, but got {x.shape[0]}."
        )

    prediction = pce_model(*x)

    return float(prediction)


# =============================================================================
# 3. Local sensitivity functions
# =============================================================================

def compute_pce_analytic_gradient(
    pce_model,
    base_composition_wt: Sequence[float],
    elements: Tuple[str, ...] = ELEMENTS,
) -> Dict[str, float]:
    """
    Compute local analytical gradients from the PCE surrogate.

    The gradient is calculated as:

        dY/dx_i

    where:
        Y   = predicted yield strength, MPa
        x_i = composition of element i, wt.%

    Parameters
    ----------
    pce_model
        Trained Chaospy PCE surrogate model.
    base_composition_wt : sequence of float
        Base composition in wt.%.
    elements : tuple of str
        Element order.

    Returns
    -------
    gradients : dict
        Dictionary containing dY/dx for each element in MPa per wt.%.
    """

    x = np.asarray(base_composition_wt, dtype=float)

    if x.shape[0] != len(elements):
        raise ValueError(
            f"base_composition_wt must have length {len(elements)}, "
            f"but got {x.shape[0]}."
        )

    gradients = {}

    for index, element in enumerate(elements):
        derivative_model = cp.derivative(pce_model, index)
        gradients[element] = float(derivative_model(*x))

    return gradients


def compute_strength_change(
    gradients: Dict[str, float],
    delta_composition_wt: Sequence[float],
    elements: Tuple[str, ...] = ELEMENTS,
    combine: str = "sum",
) -> Tuple[Dict[str, float], float]:
    """
    Estimate local strength change from composition perturbations.

    For each element:

        ΔY_i = (dY/dx_i) Δx_i

    Parameters
    ----------
    gradients : dict
        Local gradients, dY/dx_i, in MPa per wt.%.
    delta_composition_wt : sequence of float
        Composition perturbations in wt.%.
    elements : tuple of str
        Element order.
    combine : str
        Method used to combine per-element strength changes.

        Options:
            "sum" : signed sum of all ΔY_i
            "abs" : sum of absolute values, conservative total magnitude
            "rss" : root-sum-square magnitude

    Returns
    -------
    per_element_change : dict
        Per-element predicted strength changes in MPa.
    total_change : float
        Combined strength change in MPa.
    """

    delta = np.asarray(delta_composition_wt, dtype=float)

    if delta.shape[0] != len(elements):
        raise ValueError(
            f"delta_composition_wt must have length {len(elements)}, "
            f"but got {delta.shape[0]}."
        )

    per_element_change = {}

    for index, element in enumerate(elements):
        per_element_change[element] = gradients[element] * delta[index]

    values = np.array(
        [per_element_change[element] for element in elements],
        dtype=float,
    )

    if combine == "sum":
        total_change = float(np.sum(values))
    elif combine == "abs":
        total_change = float(np.sum(np.abs(values)))
    elif combine == "rss":
        total_change = float(np.sqrt(np.sum(values**2)))
    else:
        raise ValueError("combine must be one of: 'sum', 'abs', or 'rss'.")

    return per_element_change, total_change


def normalize_sensitivity(
    gradients: Dict[str, float],
    elements: Tuple[str, ...] = ELEMENTS,
) -> Dict[str, float]:
    """
    Normalize the absolute local sensitivities to the range [0, 1].

    This is useful for ranking the relative importance of elements at the
    selected base composition.

    Parameters
    ----------
    gradients : dict
        Local gradients in MPa per wt.%.
    elements : tuple of str
        Element order.

    Returns
    -------
    normalized : dict
        Normalized absolute sensitivities.
    """

    values = np.array([abs(gradients[element]) for element in elements], dtype=float)
    maximum = np.max(values)

    if maximum <= 0:
        return {element: 0.0 for element in elements}

    return {
        element: float(abs(gradients[element]) / maximum)
        for element in elements
    }


# =============================================================================
# 4. Results and plotting
# =============================================================================

def create_results_dataframe(
    elements: Tuple[str, ...],
    gradients: Dict[str, float],
    normalized_sensitivity: Dict[str, float],
    delta_composition: Sequence[float],
    strength_change: Dict[str, float],
) -> pd.DataFrame:
    """
    Create a ranked summary table for local sensitivity results.

    Returns
    -------
    results_df : pandas.DataFrame
        Ranked local sensitivity table.
    """

    results_df = pd.DataFrame({
        "Element": list(elements),
        "dY/dx (MPa/wt.%)": [gradients[element] for element in elements],
        "|dY/dx| (MPa/wt.%)": [abs(gradients[element]) for element in elements],
        "Normalized sensitivity (0-1)": [
            normalized_sensitivity[element] for element in elements
        ],
        "Delta composition (wt.%)": list(delta_composition),
        "Delta Y (MPa)": [strength_change[element] for element in elements],
    })

    results_df = results_df.sort_values(
        by="Normalized sensitivity (0-1)",
        ascending=False,
    ).reset_index(drop=True)

    return results_df


def plot_local_sensitivity(
    results_df: pd.DataFrame,
    output_dir: Path,
    show_plot: bool = True,
) -> None:
    """
    Plot normalized local sensitivity ranking.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Ranked sensitivity results.
    output_dir : Path
        Folder where the figure should be saved.
    show_plot : bool
        Whether to display the figure.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        results_df["Element"],
        results_df["Normalized sensitivity (0-1)"],
        edgecolor="black",
    )

    ax.set_xlabel("Element")
    ax.set_ylabel("Normalized local sensitivity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Local sensitivity ranking from PCE gradient")

    plt.tight_layout()

    save_path = output_dir / "local_sensitivity_ranking.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved local sensitivity plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_strength_change(
    results_df: pd.DataFrame,
    output_dir: Path,
    show_plot: bool = True,
) -> None:
    """
    Plot predicted per-element strength changes due to local perturbations.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Ranked sensitivity results.
    output_dir : Path
        Folder where the figure should be saved.
    show_plot : bool
        Whether to display the figure.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        results_df["Element"],
        results_df["Delta Y (MPa)"],
        edgecolor="black",
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Element")
    ax.set_ylabel("Estimated strength change, ΔY (MPa)")
    ax.set_title("Predicted local strength change from composition perturbation")

    plt.tight_layout()

    save_path = output_dir / "local_strength_change.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved strength-change plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# 5. Command-line interface
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser for the script.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute local sensitivity of Alloy 718 yield strength using "
            "a trained PCE surrogate model."
        )
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to the trained PCE surrogate model pickle file. "
            "Example: ../Model/PCE_surrogate_model.pkl"
        ),
    )

    parser.add_argument(
        "--base_composition",
        type=str,
        default=None,
        help=(
            "Base composition in wt.%% as a comma-separated string. "
            'Example: "Ni:52,Cr:19,Nb:5,Al:0.5,Ti:1.0,Co:0.5,Mo:3.2". '
            "If not provided, the default example composition is used."
        ),
    )

    parser.add_argument(
        "--delta_composition",
        type=str,
        default=None,
        help=(
            "Element-wise perturbation in wt.%%. Signed values are allowed. "
            'Example: "Ni:0.1,Cr:0.1,Nb:0.05,Al:0.05,Ti:-0.05,Co:0.05,Mo:0.1". '
            "If not provided, the default example perturbation is used."
        ),
    )

    parser.add_argument(
        "--combine",
        type=str,
        default="sum",
        choices=["sum", "abs", "rss"],
        help=(
            "How to combine per-element ΔY values. "
            "'sum' gives the signed total, 'abs' gives conservative magnitude, "
            "and 'rss' gives root-sum-square magnitude."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../Results/local_sensitivity",
        help="Folder where results and plots will be saved.",
    )

    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Use this flag to save figures without displaying them.",
    )

    return parser


# =============================================================================
# 6. Main workflow
# =============================================================================

def main() -> None:
    """
    Run local sensitivity analysis.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading trained PCE surrogate model...")
    pce_model = load_pce_model(args.model_path)

    if args.base_composition is None:
        base_composition_dict = DEFAULT_BASE_COMPOSITION.copy()
        print("\nNo base composition provided. Using default example composition.")
    else:
        base_composition_dict = parse_composition_string(args.base_composition)

    if args.delta_composition is None:
        delta_composition_dict = DEFAULT_DELTA_COMPOSITION.copy()
        print("\nNo perturbation composition provided. Using default example perturbation.")
    else:
        delta_composition_dict = parse_composition_string(args.delta_composition)

    base_composition = composition_dict_to_array(base_composition_dict)
    delta_composition = composition_dict_to_array(delta_composition_dict)

    print("\nBase composition, wt.%:")
    for element, value in zip(ELEMENTS, base_composition):
        print(f"  {element}: {value:.4f}")

    print("\nComposition perturbation, wt.%:")
    for element, value in zip(ELEMENTS, delta_composition):
        print(f"  Δ{element}: {value:.4f}")

    print("\nPredicting yield strength at the base composition...")
    base_prediction = predict_with_pce(pce_model, base_composition)
    print(f"  Base predicted yield strength: {base_prediction:.3f} MPa")

    print("\nComputing analytical PCE gradients...")
    gradients = compute_pce_analytic_gradient(
        pce_model=pce_model,
        base_composition_wt=base_composition,
        elements=ELEMENTS,
    )

    normalized_sensitivity = normalize_sensitivity(
        gradients=gradients,
        elements=ELEMENTS,
    )

    print("\nEstimating local strength change from perturbations...")
    strength_change, total_change = compute_strength_change(
        gradients=gradients,
        delta_composition_wt=delta_composition,
        elements=ELEMENTS,
        combine=args.combine,
    )

    results_df = create_results_dataframe(
        elements=ELEMENTS,
        gradients=gradients,
        normalized_sensitivity=normalized_sensitivity,
        delta_composition=delta_composition,
        strength_change=strength_change,
    )

    results_path = output_dir / "local_sensitivity_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nLocal sensitivity results:")
    print(results_df.to_string(index=False))

    print(f"\nTotal estimated strength change using '{args.combine}' combination:")
    print(f"  ΔY_total = {total_change:.3f} MPa")

    print(f"\nSaved local sensitivity results to: {results_path}")

    show_plot = not args.no_show

    plot_local_sensitivity(
        results_df=results_df,
        output_dir=output_dir,
        show_plot=show_plot,
    )

    plot_strength_change(
        results_df=results_df,
        output_dir=output_dir,
        show_plot=show_plot,
    )

    print("\nLocal sensitivity analysis completed.")


if __name__ == "__main__":
    main()
