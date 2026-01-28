"""
Inverse design for AM Alloy 718 using a trained PCE surrogate + Bayesian optimization (BayBE).

Script outline:
  1) Load dataset and the trained surrogate model
  2) Define helper functions:
       - bounds handling / range sampling
       - QoI prediction (mean, compositional variation, total uncertainty)
       - coverage + percentile lower bound metrics
  3) Define Bayesian optimization components:
       - search space (min/max bounds per element)
       - constraints (max >= min + epsilon)
       - objectives (uncertainty, coverage, LCB95 window)
  4) Run Bayesian optimization loop
  5) Identify Pareto front
  6) Visualize Pareto front

Notes:
- This is a pure Python script (no notebook magics).
- Assumes you already trained and saved a ChaosPy PCE surrogate as a pickle file.
- This version keeps your logic intact; adjustments are only for correctness/consistency.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection

from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe import Campaign
from baybe.recommenders import TwoPhaseMetaRecommender, RandomRecommender, BotorchRecommender
from baybe.constraints import ContinuousLinearInequalityConstraint

# NOTE: DesirabilityObjective is used below; import it explicitly
from baybe.objectives import DesirabilityObjective


# ----------------------------
# 1) Load dataset + surrogate
# ----------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)

DATASET_PATH = Path("data/Alloy718_comp_to_strength_df.csv")  # update if needed
SURROGATE_MODEL_PATH = Path("pce_surrogate.pkl")

FEATURE_COLS_WT = [
    "Ni (wt%)",
    "Cr (wt%)",
    "Nb (wt%)",
    "Al (wt%)",
    "Ti (wt%)",
    "Co (wt%)",
    "Mo (wt%)",
]

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH.resolve()}")

if not SURROGATE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Surrogate model file not found: {SURROGATE_MODEL_PATH.resolve()}\n"
        "Train and save the PCE surrogate first."
    )

Alloy718_df = pd.read_csv(DATASET_PATH)

missing_cols = [c for c in FEATURE_COLS_WT if c not in Alloy718_df.columns]
if missing_cols:
    raise ValueError(f"Dataset is missing required columns: {missing_cols}")

with open(SURROGATE_MODEL_PATH, "rb") as f:
    surrogate_model = pickle.load(f)


def surrogate_predict(X: np.ndarray) -> np.ndarray:
    """
    Predict yield strength (MPa) from compositions X of shape (N, 7) in wt%.
    Expected order: [Ni, Cr, Nb, Al, Ti, Co, Mo]
    """
    X = np.asarray(X, dtype=float)
    return surrogate_model(*X.T)


# ---------------------------------------------------------
# 2) Helper functions: bounds / sampling / QoI calculations
# ---------------------------------------------------------
def _wilson_lower(p_hat: float, n: int, conf: float = 0.95) -> float:
    """Wilson score lower bound for a binomial proportion (95% CI by default)."""
    if n <= 0:
        return 0.0
    z = 1.959963984540054  # ~N^-1(0.975)
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    radius = (z / denom) * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    return max(0.0, center - radius)


def _uniform_range(lo: float, hi: float, n: int) -> np.ndarray:
    """Sample uniformly from [lo, hi]."""
    return np.random.uniform(float(lo), float(hi), int(n))


def suggest_range_with_coverage(
    dataframe: pd.DataFrame,
    num_samples: int,
    model,
    sigma_model: float,
    target_YS: float = 1000.0,
    z_inner: float = 1.645,
) -> Tuple[list, list, list, list, list, list]:
    """
    For each candidate box (row with *_min/*_max for ALL 7 elements), sample compositions,
    predict strength, and compute:

      - mean strength over sampled compositions
      - compositional variation (std of μ(x) across samples)
      - total uncertainty: sqrt(std(μ(x))^2 + sigma_model^2)
      - "Percentile LB over box (5%)": 5th percentile of (μ(x) - z_inner*sigma_model)
      - coverage p_x: fraction of samples with (μ(x) - z_inner*sigma_model >= target)
      - coverage Wilson lower bound (95% CI)
    """
    required_cols = [
        "Ni_min", "Ni_max",
        "Cr_min", "Cr_max",
        "Nb_min", "Nb_max",
        "Al_min", "Al_max",
        "Ti_min", "Ti_max",
        "Co_min", "Co_max",
        "Mo_min", "Mo_max",
    ]
    missing = [c for c in required_cols if c not in dataframe.columns]
    if missing:
        raise KeyError(f"Candidate dataframe is missing required columns: {missing}")

    mean_strength, sigma_prop, sigma_total = [], [], []
    lb95_over_box, coverage_prop, coverage_lb = [], [], []

    for _, row in dataframe.iterrows():
        Ni = _uniform_range(row["Ni_min"], row["Ni_max"], num_samples)
        Cr = _uniform_range(row["Cr_min"], row["Cr_max"], num_samples)
        Nb = _uniform_range(row["Nb_min"], row["Nb_max"], num_samples)
        Al = _uniform_range(row["Al_min"], row["Al_max"], num_samples)
        Ti = _uniform_range(row["Ti_min"], row["Ti_max"], num_samples)
        Co = _uniform_range(row["Co_min"], row["Co_max"], num_samples)
        Mo = _uniform_range(row["Mo_min"], row["Mo_max"], num_samples)

        X = np.vstack([Ni, Cr, Nb, Al, Ti, Co, Mo]).T  # (N, 7)
        mu_i = model(*X.T)

        # Per-composition lower bound check
        lb_i = mu_i - z_inner * float(sigma_model) - target_YS
        safe = (lb_i >= 0.0)
        p_hat = float(np.mean(safe))
        p_lb = _wilson_lower(p_hat, len(mu_i), conf=0.95)

        # 5th percentile LCB over the box (absolute MPa)
        q05_lb = float(np.quantile(mu_i - z_inner * float(sigma_model), 0.05))

        # "Total uncertainty": composition spread + non-compositional term
        s_comp = float(np.std(mu_i, ddof=1))
        s_tot = math.sqrt(s_comp**2 + float(sigma_model) ** 2)

        mean_strength.append(float(np.mean(mu_i)))
        sigma_prop.append(s_comp)
        sigma_total.append(s_tot)
        lb95_over_box.append(q05_lb)
        coverage_prop.append(p_hat)
        coverage_lb.append(p_lb)

    return (
        mean_strength,
        sigma_prop,
        sigma_total,
        lb95_over_box,
        coverage_prop,
        coverage_lb,
    )


# ---------------------------------------------------------
# 3) Bayesian optimization: space, constraints, objectives
# ---------------------------------------------------------
def build_searchspace(df: pd.DataFrame) -> SearchSpace:
    """
    Create the BayBE search space for min/max bounds of each element.
    Bounds are set from the observed dataset ranges.
    """
    def _bounds(col: str) -> Tuple[float, float]:
        return (float(df[col].min()), float(df[col].max()))

    cont_params_7 = [
        NumericalContinuousParameter("Ni_min", bounds=_bounds("Ni (wt%)")),
        NumericalContinuousParameter("Ni_max", bounds=_bounds("Ni (wt%)")),
        NumericalContinuousParameter("Cr_min", bounds=_bounds("Cr (wt%)")),
        NumericalContinuousParameter("Cr_max", bounds=_bounds("Cr (wt%)")),
        NumericalContinuousParameter("Nb_min", bounds=_bounds("Nb (wt%)")),
        NumericalContinuousParameter("Nb_max", bounds=_bounds("Nb (wt%)")),
        NumericalContinuousParameter("Al_min", bounds=_bounds("Al (wt%)")),
        NumericalContinuousParameter("Al_max", bounds=_bounds("Al (wt%)")),
        NumericalContinuousParameter("Ti_min", bounds=_bounds("Ti (wt%)")),
        NumericalContinuousParameter("Ti_max", bounds=_bounds("Ti (wt%)")),
        NumericalContinuousParameter("Co_min", bounds=_bounds("Co (wt%)")),
        NumericalContinuousParameter("Co_max", bounds=_bounds("Co (wt%)")),
        NumericalContinuousParameter("Mo_min", bounds=_bounds("Mo (wt%)")),
        NumericalContinuousParameter("Mo_max", bounds=_bounds("Mo (wt%)")),
    ]

    # Enforce max >= min + eps (eps prevents degenerate zero-width ranges)
    eps = 0.01
    constraints_7 = [
        ContinuousLinearInequalityConstraint(parameters=["Ni_max", "Ni_min"], coefficients=[1, -1], rhs=eps),
        ContinuousLinearInequalityConstraint(parameters=["Cr_max", "Cr_min"], coefficients=[1, -1], rhs=eps),
        ContinuousLinearInequalityConstraint(parameters=["Nb_max", "Nb_min"], coefficients=[1, -1], rhs=eps),
        ContinuousLinearInequalityConstraint(parameters=["Al_max", "Al_min"], coefficients=[1, -1], rhs=eps),
        ContinuousLinearInequalityConstraint(parameters=["Ti_max", "Ti_min"], coefficients=[1, -1], rhs=eps),
        ContinuousLinearInequalityConstraint(parameters=["Co_max", "Co_min"], coefficients=[1, -1], rhs=eps),
        ContinuousLinearInequalityConstraint(parameters=["Mo_max", "Mo_min"], coefficients=[1, -1], rhs=eps),
    ]

    return SearchSpace.from_product(parameters=cont_params_7, constraints=constraints_7)


def build_campaign(space: SearchSpace) -> Campaign:
    """
    Define multi-objective desirability objective + hybrid recommender (random -> BO).
    """
    obj_1 = NumericalTarget(
        name="Total uncertainty",
        mode="MIN",
        bounds=(0, 45),
        transformation="LINEAR",
    )
    obj_2 = NumericalTarget(
        name="Coverage over box p_x",
        mode="MATCH",
        bounds=(0.95, 1.0),
        transformation="TRIANGULAR",
    )
    obj_3 = NumericalTarget(
        name="Percentile LB over box (5%) (MPa)",
        mode="MATCH",
        bounds=(1000.0, 1020.0),
        transformation="BELL",
    )

    objective = DesirabilityObjective(
        targets=[obj_1, obj_2, obj_3],
        weights=[0.5, 0.3, 0.2],
        scalarizer="GEOM_MEAN",
    )

    recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(acquisition_function="qLogEI"),
    )

    return Campaign(searchspace=space, objective=objective, recommender=recommender)


# ----------------------------
# 4) Run Bayesian optimization
# ----------------------------
def run_bayesian_optimization(
    campaign: Campaign,
    model,
    n_iters: int = 25,
    batch_size: int = 10,
    num_mc_samples: int = 1000,
    target_YS: float = 1000.0,
    sigma_model: float = 32.417,
    z_inner: float = 1.645,
    lb_window: tuple[float, float] = (1000.0, 1020.0),
    sigma_max: float = 45.0,
    coverage_min: float | None = 0.95,  # set to None to skip coverage filtering
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      full_data: all evaluated candidates across all BO iterations
      filtered_data: subset satisfying the design requirements
    """
    for k in range(n_iters):
        print(f"Running BO iteration {k + 1}/{n_iters}")

        iter_df = campaign.recommend(batch_size=batch_size)

        (mu, s_comp, s_tot, lb5, cover_p, cover_lb) = suggest_range_with_coverage(
            dataframe=iter_df,
            num_samples=num_mc_samples,
            model=model,
            sigma_model=sigma_model,
            target_YS=target_YS,
            z_inner=z_inner,
        )

        iter_df["Predicted yield strength (MPa)"] = mu
        iter_df["Compositional variation"] = s_comp
        iter_df["Total uncertainty"] = s_tot
        iter_df["Percentile LB over box (5%) (MPa)"] = lb5
        iter_df["Coverage over box p_x"] = cover_p
        iter_df["Coverage over box LB (95% CI)"] = cover_lb

        campaign.add_measurements(iter_df)

    full_data = campaign.measurements.copy()

    mask = (
        full_data["Percentile LB over box (5%) (MPa)"].between(lb_window[0], lb_window[1], inclusive="both")
        & (full_data["Total uncertainty"] <= sigma_max)
    )
    if coverage_min is not None:
        mask &= (full_data["Coverage over box p_x"] >= coverage_min)

    filtered_data = (
        full_data.loc[mask]
        .sort_values("Total uncertainty", ascending=True)
        .reset_index(drop=True)
    )

    return full_data, filtered_data


# ----------------------------
# 5) Pareto front analysis
# ----------------------------
def pareto_front_2d(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    maximize_x: bool = True,
    eps_x: float = 0.0,  # kept for consistency
    eps_y: float = 0.2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Pareto front extraction for 2 objectives:
      - maximize xcol
      - minimize ycol

    Assumes df is already filtered to only include acceptable solutions.
    """
    if df.empty:
        return df.copy(), np.zeros(0, dtype=bool)

    work = df[[xcol, ycol]].copy()
    work[xcol] = pd.to_numeric(work[xcol], errors="coerce")
    work[ycol] = pd.to_numeric(work[ycol], errors="coerce")
    valid = work.dropna()

    if valid.empty:
        return df.iloc[0:0].copy(), np.zeros(len(df), dtype=bool)

    arr = valid.to_numpy(dtype=float)

    order = np.argsort(-arr[:, 0]) if maximize_x else np.argsort(arr[:, 0])
    y_s = arr[order, 1]

    keep = []
    best_y = np.inf
    for k, idx in enumerate(order):
        if y_s[k] < best_y - eps_y:
            keep.append(idx)
            best_y = y_s[k]

    # keep contains positions in valid, map back to df mask
    pareto_valid_index = valid.index[np.array(keep, dtype=int)]
    mask = df.index.isin(pareto_valid_index).to_numpy()

    front = df.loc[mask].copy().sort_values(by=xcol, ascending=True)
    return front, mask


def knee_on_polyline(x: Iterable[float], y: Iterable[float]) -> int | None:
    """Return index of 'knee' point using max distance to chord heuristic."""
    x = np.asarray(list(x), float)
    y = np.asarray(list(y), float)
    if len(x) < 3:
        return None
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0) + 1e-12
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return int(np.argmax(d))


def _trim_segment(ax, p0, p1, gap_pts: float = 6.0):
    """Trim polyline segments so lines do not run under marker circles."""
    trans = ax.transData
    inv = trans.inverted()
    P0 = trans.transform(p0)
    P1 = trans.transform(p1)
    v = P1 - P0
    L = np.hypot(v[0], v[1])
    if L <= 2 * gap_pts:
        return None
    u = v / L
    Q0 = P0 + u * gap_pts
    Q1 = P1 - u * gap_pts
    return inv.transform(Q0), inv.transform(Q1)


# ----------------------------
# 6) Visualization
# ----------------------------
def plot_pareto_front(
    df_all: pd.DataFrame,
    pareto_df: pd.DataFrame,
    xcol: str,
    ycol: str,
    cov_col: str,
    cov_min: float = 0.95,
    color_by: str | None = None,
):
    plt.rcParams["font.family"] = "calibri"

    dfc = df_all.copy()
    if (cov_col in dfc.columns) and (cov_min is not None):
        dfc = dfc[dfc[cov_col] >= cov_min].copy()

    if dfc.empty:
        print("plot_pareto_front: no data to plot.")
        return

    if color_by is None:
        color_by = cov_col if cov_col in dfc.columns else xcol

    cvals = pd.to_numeric(dfc[color_by], errors="coerce").to_numpy(float)
    norm = mcolors.Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        dfc[xcol],
        dfc[ycol],
        s=150,
        c=cvals,
        cmap=cmap,
        norm=norm,
        alpha=0.60,
        linewidths=0,
        label="Filtered designs",
    )

    if not pareto_df.empty:
        ax.scatter(
            pareto_df[xcol],
            pareto_df[ycol],
            s=200,
            facecolor="none",
            edgecolor="k",
            linewidths=1.2,
            zorder=3,
            label="Pareto designs",
        )

        segments = []
        if len(pareto_df) >= 2:
            fig.canvas.draw()
            for i in range(len(pareto_df) - 1):
                p0 = (pareto_df.iloc[i][xcol], pareto_df.iloc[i][ycol])
                p1 = (pareto_df.iloc[i + 1][xcol], pareto_df.iloc[i + 1][ycol])
                trimmed = _trim_segment(ax, p0, p1, gap_pts=8.0)
                if trimmed is not None:
                    q0, q1 = trimmed
                    segments.append([q0, q1])

        if segments:
            lc = LineCollection(segments, colors=["C0"], linewidths=4, zorder=2)
            lc.set_label("Pareto front")
            ax.add_collection(lc)

        knee_idx = knee_on_polyline(pareto_df[xcol], pareto_df[ycol])
        if knee_idx is not None:
            ax.scatter(
                pareto_df.iloc[knee_idx][xcol],
                pareto_df.iloc[knee_idx][ycol],
                s=180,
                facecolor="gold",
                edgecolor="k",
                linewidths=1.2,
                zorder=4,
                label="Knee",
            )

    ax.set_xlabel("Expected strength LCB₉₅ over box (5th percentile) (MPa)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Total uncertainty (MPa)", fontsize=18, fontweight="bold")

    legend_prop = FontProperties(size=14, weight="bold")
    ax.legend(frameon=False, loc="best", prop=legend_prop)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(color_by, fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.show()


# ----------------------------
# Main runner
# ----------------------------
def main() -> None:
    space = build_searchspace(Alloy718_df)
    campaign = build_campaign(space)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df, filtered_df = run_bayesian_optimization(
        campaign=campaign,
        model=surrogate_model,
        n_iters=25,
        batch_size=10,
        num_mc_samples=1000,
        target_YS=1000.0,
        sigma_model=32.417,
        z_inner=1.645,
        lb_window=(1000.0, 1020.0),
        sigma_max=45.0,
        coverage_min=0.95,
    )

    full_df.to_csv(out_dir / "bo_measurements_full.csv", index=False)
    filtered_df.to_csv(out_dir / "bo_measurements_filtered.csv", index=False)

    if filtered_df.empty:
        print("No candidates satisfied the filter criteria. Try relaxing constraints or running more iterations.")
        return

    xcol = "Percentile LB over box (5%) (MPa)"
    ycol = "Total uncertainty"
    cov_col = "Coverage over box p_x"

    pareto_df, _ = pareto_front_2d(filtered_df, xcol=xcol, ycol=ycol, maximize_x=True, eps_x=0.5, eps_y=0.2)
    pareto_df.to_csv(out_dir / "pareto_front_filtered.csv", index=False)

    plot_pareto_front(
        df_all=filtered_df,
        pareto_df=pareto_df,
        xcol=xcol,
        ycol=ycol,
        cov_col=cov_col,
        cov_min=0.95,
        color_by=cov_col,
    )

    print(f"Saved: {out_dir / 'bo_measurements_full.csv'}")
    print(f"Saved: {out_dir / 'bo_measurements_filtered.csv'}")
    print(f"Saved: {out_dir / 'pareto_front_filtered.csv'}")


if __name__ == "__main__":
    main()
