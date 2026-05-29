from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baybe import Campaign
from baybe.constraints import ContinuousLinearInequalityConstraint
from baybe.objective import DesirabilityObjective
from baybe.parameters import NumericalContinuousParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

np.random.seed(42)

ELEMENTS: Tuple[str, ...] = ("Ni", "Cr", "Nb", "Al", "Ti", "Co", "Mo")
OPTIMIZED_ELEMENTS: Tuple[str, ...] = ("Nb", "Al", "Ti")
BACKGROUND_ELEMENTS: Tuple[str, ...] = ("Ni", "Cr", "Co", "Mo")

DATASET_PATH = Path("../data/example_alloy718_dataset.csv")
SURROGATE_MODEL_PATH = Path("../model/pce_surrogate.pkl")

DEFAULT_NUM_MC_SAMPLES = 1000
DEFAULT_SIGMA_MODEL = 32.417
DEFAULT_Z_INNER = 1.645

DEFAULT_RANGES_WT = {
    "Ni": (50.0, 55.0),
    "Cr": (17.0, 21.0),
    "Nb": (4.75, 5.50),
    "Al": (0.20, 0.80),
    "Ti": (0.65, 1.15),
    "Co": (0.00, 1.00),
    "Mo": (2.80, 3.30),
}

OUTPUT_DIR = Path("outputs_3element")


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------

def load_dataset_for_bounds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path.resolve()}\n"
            "Update DATASET_PATH to your local dataset."
        )
    return pd.read_csv(path)


def load_surrogate_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Surrogate model file not found: {path.resolve()}\n"
            "Update SURROGATE_MODEL_PATH to your trained PCE model."
        )

    with open(path, "rb") as file:
        return pickle.load(file)


# ---------------------------------------------------------
# Sampling and reliability metrics
# ---------------------------------------------------------

def _wilson_lower_95(p_hat: float, n: int) -> float:
    if n <= 0:
        return 0.0

    z = 1.959963984540054
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    radius = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) + z**2 / (4 * n)) / n
    )

    return max(0.0, center - radius)


def _uniform_range(lo: float, hi: float, n: int) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)

    if hi < lo:
        lo, hi = hi, lo

    return np.random.uniform(lo, hi, int(n))


def suggest_range_with_coverage_3elem(
    dataframe: pd.DataFrame,
    num_samples: int,
    model,
    sigma_model: float,
    target_YS: float,
    z_inner: float = DEFAULT_Z_INNER,
    fixed_ranges_wt: Dict[str, Tuple[float, float]] = DEFAULT_RANGES_WT,
) -> Tuple[list, list, list, list, list, list]:

    required_cols = []

    for element in OPTIMIZED_ELEMENTS:
        required_cols += [f"{element}_min", f"{element}_max"]

    missing = [col for col in required_cols if col not in dataframe.columns]

    if missing:
        raise KeyError(f"Candidate DataFrame is missing columns: {missing}")

    mean_strength = []
    sigma_comp = []
    sigma_total = []
    lb05_over_box = []
    coverage_prop = []
    coverage_lb = []

    sigma_model = float(sigma_model)
    target_YS = float(target_YS)

    for _, row in dataframe.iterrows():
        Nb = _uniform_range(row["Nb_min"], row["Nb_max"], num_samples)
        Al = _uniform_range(row["Al_min"], row["Al_max"], num_samples)
        Ti = _uniform_range(row["Ti_min"], row["Ti_max"], num_samples)

        Ni = _uniform_range(*fixed_ranges_wt["Ni"], num_samples)
        Cr = _uniform_range(*fixed_ranges_wt["Cr"], num_samples)
        Co = _uniform_range(*fixed_ranges_wt["Co"], num_samples)
        Mo = _uniform_range(*fixed_ranges_wt["Mo"], num_samples)

        X = np.vstack([Ni, Cr, Nb, Al, Ti, Co, Mo]).T

        mu_i = np.asarray(model(*X.T), dtype=float).reshape(-1)

        lcb_i = mu_i - z_inner * sigma_model
        safe = lcb_i >= target_YS

        p_hat = float(np.mean(safe))
        p_lb = _wilson_lower_95(p_hat, len(mu_i))

        q05_lb = float(np.quantile(lcb_i, 0.05))
        s_comp = float(np.std(mu_i, ddof=1))
        s_total = math.sqrt(s_comp**2 + sigma_model**2)

        mean_strength.append(float(np.mean(mu_i)))
        sigma_comp.append(s_comp)
        sigma_total.append(s_total)
        lb05_over_box.append(q05_lb)
        coverage_prop.append(p_hat)
        coverage_lb.append(p_lb)

    return (
        mean_strength,
        sigma_comp,
        sigma_total,
        lb05_over_box,
        coverage_prop,
        coverage_lb,
    )


# ---------------------------------------------------------
# BayBE search space and campaign
# ---------------------------------------------------------

def build_searchspace_3elem(
    df: pd.DataFrame,
    *,
    min_deltas: Dict[str, float],
) -> SearchSpace:

    def _bounds(column: str) -> Tuple[float, float]:
        if column not in df.columns:
            raise KeyError(f"Dataset is missing required column: {column}")

        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid bounds for {column}: {lo}, {hi}")

        return lo, hi

    parameters = []

    for element in OPTIMIZED_ELEMENTS:
        lo, hi = _bounds(f"{element} (wt%)")

        parameters.append(
            NumericalContinuousParameter(
                f"{element}_min",
                bounds=(lo, hi),
            )
        )

        parameters.append(
            NumericalContinuousParameter(
                f"{element}_max",
                bounds=(lo, hi),
            )
        )

    constraints = []

    for element in OPTIMIZED_ELEMENTS:
        eps = float(min_deltas[element])

        if eps <= 0:
            raise ValueError(f"Minimum window width for {element} must be > 0.")

        constraints.append(
            ContinuousLinearInequalityConstraint(
                parameters=[f"{element}_max", f"{element}_min"],
                coefficients=[1.0, -1.0],
                rhs=eps,
            )
        )

    return SearchSpace.from_product(
        parameters=parameters,
        constraints=constraints,
    )


def build_campaign(
    space: SearchSpace,
    *,
    min_required_strength: float,
    sigma_threshold: float,
    reliability_target: float,
    weights: Tuple[float, float, float],
) -> Campaign:

    min_required_strength = float(min_required_strength)
    sigma_threshold = float(sigma_threshold)
    reliability_target = float(reliability_target)

    if sigma_threshold <= 0:
        raise ValueError("sigma_threshold must be > 0.")

    if not (0.0 < reliability_target <= 1.0):
        raise ValueError("reliability_target must be in (0, 1].")

    w1, w2, w3 = map(float, weights)
    weight_sum = w1 + w2 + w3

    if weight_sum <= 0:
        raise ValueError("Objective weights must sum to a positive value.")

    weights_norm = [
        w1 / weight_sum,
        w2 / weight_sum,
        w3 / weight_sum,
    ]

    lb_lo = min_required_strength
    lb_hi = min_required_strength + 0.5 * sigma_threshold

    target_uncertainty = NumericalTarget(
        name="Total uncertainty",
        mode="MIN",
        bounds=(0.0, sigma_threshold),
        transformation="LINEAR",
    )

    target_coverage = NumericalTarget(
        name="Coverage over box p_x",
        mode="MATCH",
        bounds=(reliability_target, 1.0),
        transformation="TRIANGULAR",
    )

    target_strength = NumericalTarget(
        name="Percentile LB over box (5%) (MPa)",
        mode="MATCH",
        bounds=(lb_lo, lb_hi),
        transformation="BELL",
    )

    objective = DesirabilityObjective(
        targets=[
            target_uncertainty,
            target_coverage,
            target_strength,
        ],
        weights=weights_norm,
        scalarizer="GEOM_MEAN",
    )

    recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(acquisition_function="qLogEI"),
    )

    return Campaign(
        searchspace=space,
        objective=objective,
        recommender=recommender,
    )


# ---------------------------------------------------------
# Bayesian optimization
# ---------------------------------------------------------

def run_bayesian_optimization_3elem(
    campaign: Campaign,
    model,
    *,
    n_iters: int,
    batch_size: int,
    num_mc_samples: int,
    target_YS: float,
    sigma_model: float,
    z_inner: float,
    lb_window: Tuple[float, float],
    sigma_max: float,
    coverage_min: Optional[float],
    fixed_ranges_wt: Dict[str, Tuple[float, float]] = DEFAULT_RANGES_WT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if n_iters <= 0:
        raise ValueError("n_iters must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_mc_samples <= 0:
        raise ValueError("num_mc_samples must be positive.")

    for iteration in range(n_iters):
        print(f"Running BO iteration {iteration + 1}/{n_iters}")

        iter_df = campaign.recommend(batch_size=batch_size)

        (
            mean_strength,
            s_comp,
            s_total,
            lb5,
            coverage_p,
            coverage_lb,
        ) = suggest_range_with_coverage_3elem(
            dataframe=iter_df,
            num_samples=num_mc_samples,
            model=model,
            sigma_model=sigma_model,
            target_YS=target_YS,
            z_inner=z_inner,
            fixed_ranges_wt=fixed_ranges_wt,
        )

        for element in BACKGROUND_ELEMENTS:
            iter_df[f"{element}_min"] = fixed_ranges_wt[element][0]
            iter_df[f"{element}_max"] = fixed_ranges_wt[element][1]

        iter_df["Predicted yield strength (MPa)"] = mean_strength
        iter_df["Compositional variation"] = s_comp
        iter_df["Total uncertainty"] = s_total
        iter_df["Percentile LB over box (5%) (MPa)"] = lb5
        iter_df["Coverage over box p_x"] = coverage_p
        iter_df["Coverage over box LB (95% CI)"] = coverage_lb

        campaign.add_measurements(iter_df)

    full_data = campaign.measurements.copy()

    mask = (
        full_data["Percentile LB over box (5%) (MPa)"].between(
            lb_window[0],
            lb_window[1],
            inclusive="both",
        )
        & (full_data["Total uncertainty"] <= float(sigma_max))
    )

    if coverage_min is not None:
        mask &= full_data["Coverage over box p_x"] >= float(coverage_min)

    filtered_data = (
        full_data.loc[mask]
        .sort_values("Total uncertainty", ascending=True)
        .reset_index(drop=True)
    )

    return full_data, filtered_data


# ---------------------------------------------------------
# Pareto filtering
# ---------------------------------------------------------

def pareto_front_2d(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    maximize_x: bool = True,
    eps_y: float = 0.2,
) -> Tuple[pd.DataFrame, np.ndarray]:

    if df.empty:
        return df.copy(), np.zeros(0, dtype=bool)

    if xcol not in df.columns or ycol not in df.columns:
        raise KeyError(f"Missing columns: {xcol}, {ycol}")

    work = df[[xcol, ycol]].copy()
    work[xcol] = pd.to_numeric(work[xcol], errors="coerce")
    work[ycol] = pd.to_numeric(work[ycol], errors="coerce")

    valid = work.dropna()

    if valid.empty:
        return df.iloc[0:0].copy(), np.zeros(len(df), dtype=bool)

    arr = valid.to_numpy(dtype=float)

    order = np.argsort(-arr[:, 0]) if maximize_x else np.argsort(arr[:, 0])
    y_sorted = arr[order, 1]

    keep_pos = []
    best_y = np.inf

    for k, pos in enumerate(order):
        if y_sorted[k] < best_y - float(eps_y):
            keep_pos.append(pos)
            best_y = y_sorted[k]

    pareto_index = valid.index[np.array(keep_pos, dtype=int)]
    mask = np.asarray(df.index.isin(pareto_index), dtype=bool)

    front = df.loc[mask].copy().sort_values(by=xcol, ascending=True)

    return front, mask


# ---------------------------------------------------------
# Plot selected design window
# ---------------------------------------------------------

def plot_design_window(
    row: pd.Series,
    *,
    pareto_df: pd.DataFrame,
    target_YS: float,
    out_path: Path | None = None,
    ranges_wt: Dict[str, Tuple[float, float]] = DEFAULT_RANGES_WT,
    strength_col: str = "Percentile LB over box (5%) (MPa)",
    coverage_col: str = "Coverage over box p_x",
) -> plt.Figure:

    if coverage_col not in pareto_df.columns:
        raise KeyError(f"Coverage column not found: {coverage_col}")

    for element in ELEMENTS:
        if f"{element}_min" not in row.index or f"{element}_max" not in row.index:
            raise KeyError(f"Missing min/max columns for {element}")

    color_values = pd.to_numeric(
        pareto_df[strength_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    vmax = float(np.nanmax(color_values))
    vmin = float(target_YS)

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.coolwarm

    axis_labels = list(ELEMENTS) + ["Reliability"]
    x = np.arange(len(axis_labels))

    fig, ax = plt.subplots(figsize=(12, 6))

    y_min = []
    y_max = []

    for element in ELEMENTS:
        lo, hi = ranges_wt[element]

        v_min = float(row[f"{element}_min"])
        v_max = float(row[f"{element}_max"])

        y_min.append((v_min - lo) / (hi - lo + 1e-12))
        y_max.append((v_max - lo) / (hi - lo + 1e-12))

    cov_values = pd.to_numeric(
        pareto_df[coverage_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    cov_lo = float(np.nanmin(cov_values))
    cov_hi = float(np.nanmax(cov_values))

    if not np.isfinite(cov_lo) or not np.isfinite(cov_hi) or cov_hi <= cov_lo:
        cov_lo, cov_hi = 0.0, 1.0

    cov = float(row[coverage_col])
    y_cov = (cov - cov_lo) / (cov_hi - cov_lo + 1e-12)

    y_min.append(y_cov)
    y_max.append(y_cov)

    y_min = np.asarray(y_min, dtype=float)
    y_max = np.asarray(y_max, dtype=float)

    y_lo = np.minimum(y_min, y_max)
    y_hi = np.maximum(y_min, y_max)

    color = cmap(norm(float(row[strength_col])))

    ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.30)
    ax.plot(x, y_lo, color=color, linewidth=1.2)
    ax.plot(x, y_hi, color=color, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(axis_labels, fontsize=13, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.25, len(axis_labels) - 0.75)

    for xi in x:
        ax.vlines(xi, 0, 1, color="black", linewidth=1.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label("Expected LCB95 strength (MPa)", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------
# User prompts
# ---------------------------------------------------------

def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(
            f"{prompt}" + (f" [default={default}]: " if default is not None else ": ")
        ).strip()

        if raw == "" and default is not None:
            return float(default)

        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def _prompt_int(prompt: str, default: int | None = None) -> int:
    while True:
        raw = input(
            f"{prompt}" + (f" [default={default}]: " if default is not None else ": ")
        ).strip()

        if raw == "" and default is not None:
            return int(default)

        try:
            value = int(raw)

            if value <= 0:
                print("Please enter a positive integer.")
                continue

            return value

        except ValueError:
            print("Please enter a valid integer.")


def _prompt_weights(
    default: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> Tuple[float, float, float]:

    while True:
        raw = input(f"Enter objective weights w1,w2,w3 [default={default}]: ").strip()

        if raw == "":
            return tuple(map(float, default))

        parts = [part.strip() for part in raw.split(",")]

        if len(parts) != 3:
            print("Enter exactly 3 values, for example: 0.5,0.3,0.2")
            continue

        try:
            weights = tuple(float(part) for part in parts)

            if sum(weights) <= 0:
                print("Weights must sum to a positive value.")
                continue

            return weights

        except ValueError:
            print("Please enter valid numbers.")


def _prompt_min_deltas_3elem() -> Dict[str, float]:
    while True:
        raw = input(
            "Minimum window width for Nb,Al,Ti. Enter one value for all or "
            "3 comma-separated values [default=0.01]: "
        ).strip()

        if raw == "":
            return {element: 0.01 for element in OPTIMIZED_ELEMENTS}

        parts = [part.strip() for part in raw.split(",") if part.strip()]

        try:
            if len(parts) == 1:
                value = float(parts[0])

                if value <= 0:
                    print("Value must be > 0.")
                    continue

                return {element: value for element in OPTIMIZED_ELEMENTS}

            if len(parts) == len(OPTIMIZED_ELEMENTS):
                values = [float(part) for part in parts]

                if any(value <= 0 for value in values):
                    print("All values must be > 0.")
                    continue

                return {
                    element: value
                    for element, value in zip(OPTIMIZED_ELEMENTS, values)
                }

            print("Enter either 1 value or 3 comma-separated values.")

        except ValueError:
            print("Please enter valid numbers.")


def _prompt_plot_row_index(n_rows: int) -> Optional[int]:
    while True:
        raw = input(
            f"\nEnter a filtered row index to plot (0..{n_rows - 1}), "
            "or press Enter to skip: "
        ).strip().lower()

        if raw in ("", "q", "quit", "esc", "escape"):
            return None

        try:
            index = int(raw)

            if 0 <= index < n_rows:
                return index

            print("Index out of range.")

        except ValueError:
            print("Please enter a valid index or press Enter to skip.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    df_bounds = load_dataset_for_bounds(DATASET_PATH)
    surrogate_model = load_surrogate_model(SURROGATE_MODEL_PATH)

    min_required_strength = _prompt_float(
        "Minimum required strength (MPa)",
        default=1000.0,
    )

    sigma_threshold = _prompt_float(
        "Allowable total uncertainty threshold (MPa)",
        default=45.0,
    )

    reliability_pct = _prompt_float(
        "Target reliability (%)",
        default=95.0,
    )

    reliability_target = (
        reliability_pct / 100.0
        if reliability_pct > 1.0
        else reliability_pct
    )

    if not (0.0 < reliability_target <= 1.0):
        raise ValueError("Reliability must be provided as 0-1 or 0-100%.")

    weights = _prompt_weights(default=(0.5, 0.3, 0.2))

    n_iters = _prompt_int("Number of BO iterations", default=10)
    batch_size = _prompt_int("Batch size per iteration", default=10)

    min_deltas = _prompt_min_deltas_3elem()

    num_mc_samples = DEFAULT_NUM_MC_SAMPLES
    sigma_model = DEFAULT_SIGMA_MODEL
    z_inner = DEFAULT_Z_INNER

    lb_lo = float(min_required_strength)
    lb_hi = float(min_required_strength + 0.5 * sigma_threshold)
    lb_window = (lb_lo, lb_hi)

    target_YS = float(min_required_strength)
    sigma_max = float(sigma_threshold)
    coverage_min = float(reliability_target)

    space = build_searchspace_3elem(
        df_bounds,
        min_deltas=min_deltas,
    )

    campaign = build_campaign(
        space,
        min_required_strength=min_required_strength,
        sigma_threshold=sigma_threshold,
        reliability_target=reliability_target,
        weights=weights,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_df, filtered_df = run_bayesian_optimization_3elem(
        campaign=campaign,
        model=surrogate_model,
        n_iters=n_iters,
        batch_size=batch_size,
        num_mc_samples=num_mc_samples,
        target_YS=target_YS,
        sigma_model=sigma_model,
        z_inner=z_inner,
        lb_window=lb_window,
        sigma_max=sigma_max,
        coverage_min=coverage_min,
        fixed_ranges_wt=DEFAULT_RANGES_WT,
    )

    full_path = OUTPUT_DIR / "bo_measurements_full_3element.csv"
    filtered_path = OUTPUT_DIR / "filtered_candidates_3element.csv"

    full_df.to_csv(full_path, index=False)
    filtered_df.to_csv(filtered_path, index=False)

    print(f"\nSaved full results to: {full_path}")
    print(f"Saved filtered results to: {filtered_path}")

    if filtered_df.empty:
        print("\nNo candidates satisfied the filter criteria.")
        return

    xcol = "Percentile LB over box (5%) (MPa)"
    ycol = "Total uncertainty"

    pareto_df, _ = pareto_front_2d(
        filtered_df,
        xcol=xcol,
        ycol=ycol,
        maximize_x=True,
        eps_y=0.2,
    )

    pareto_path = OUTPUT_DIR / "pareto_front_3element.csv"
    pareto_df.to_csv(pareto_path, index=False)

    print(f"Saved Pareto front to: {pareto_path}")

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)

    print("\nFiltered candidate windows:")
    print(filtered_df)

    selected_index = _prompt_plot_row_index(len(filtered_df))

    if selected_index is None:
        print("No plot requested.")
        return

    row = filtered_df.iloc[selected_index]
    figure_path = OUTPUT_DIR / f"coordinate_plot_3element_row_{selected_index}.png"

    fig = plot_design_window(
        row,
        pareto_df=pareto_df,
        target_YS=target_YS,
        out_path=figure_path,
    )

    print(f"Saved plot to: {figure_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
