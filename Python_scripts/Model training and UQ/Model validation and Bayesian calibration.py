"""
Model validation and Bayesian calibration for Alloy 718 yield strength.

Part 1: Validate PCE surrogate against experimental AM data (CSV).
Part 2: Bayesian calibration of global bias and uncertainty using PyMC.

All original columns in the experimental dataset are preserved.
"""


from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az



# ----------------------------
# Configuration
# ----------------------------

RNG_SEED = 42

# Inputs
DATA_PATH = Path("../Data/Experimental yield strength of Alloy 718.csv")

# Trained surrogate model (exported from the surrogate training script)
SURROGATE_MODEL_PATH = Path("../Model/pce_surrogate.pkl")

COMP_COLS = ["Ni", "Cr", "Nb", "Al", "Ti", "Co", "Mo"]
TARGET_COL = "Yield strength (MPa)"
NAME_COL = "Name"


# ----------------------------
# Load surrogate model first
# ----------------------------

if not SURROGATE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Surrogate model file not found: {SURROGATE_MODEL_PATH.resolve()}\n"
        "Make sure you trained and saved the PCE model first."
    )

with open(SURROGATE_MODEL_PATH, "rb") as f:
    surrogate_model = pickle.load(f)


def surrogate_predict(X: np.ndarray) -> np.ndarray:
    """Return mean prediction from the loaded PCE surrogate."""
    X = np.asarray(X, dtype=float)
    return surrogate_model(*X.T)


# ----------------------------
# Load experimental data
# ----------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Experimental CSV not found: {DATA_PATH.resolve()}")

exp_data = pd.read_csv(DATA_PATH)

missing = [c for c in COMP_COLS + [TARGET_COL] if c not in exp_data.columns]
if missing:
    raise ValueError(f"Missing required columns in experimental CSV: {missing}")

X_exp = exp_data[COMP_COLS].to_numpy(dtype=float)
y_exp = exp_data[TARGET_COL].to_numpy(dtype=float)


# ----------------------------
# Surrogate prediction + attach results (no column drops)
# ----------------------------

y_pred = surrogate_predict(X_exp)

target_idx = exp_data.columns.get_loc(TARGET_COL)
exp_data.insert(target_idx + 1, "Predicted strength (MPa)", y_pred)

# Convenience columns (optional, keeps all original columns)
exp_data["Absolute error (MPa)"] = np.abs(exp_data["Predicted strength (MPa)"] - exp_data[TARGET_COL])
exp_data["Absolute error (%)"] = 100.0 * exp_data["Absolute error (MPa)"] / exp_data[TARGET_COL]


# ----------------------------
# Visualization: validation
# ----------------------------

if NAME_COL not in exp_data.columns:
    # Fallback if "Name" doesn't exist
    names = np.arange(len(exp_data)).astype(str)
else:
    names = exp_data[NAME_COL].astype(str).values

actual = exp_data[TARGET_COL].to_numpy(dtype=float)
predicted = exp_data["Predicted strength (MPa)"].to_numpy(dtype=float)
error_percent = exp_data["Absolute error (%)"].to_numpy(dtype=float)

x = np.arange(len(names))
bar_width = 0.35
offset = 30

plt.figure(figsize=(14, 7))
plt.rcParams["font.family"] = "calibri"

plt.bar(
    x - bar_width / 2,
    actual,
    width=bar_width,
    label="Literature",
    color="#5ab4ac",
    edgecolor="black",
)
plt.bar(
    x + bar_width / 2,
    predicted,
    width=bar_width,
    label="Predicted",
    color="#fc8d62",
    edgecolor="black",
)

for i in range(len(names)):
    top = max(actual[i], predicted[i])
    plt.text(
        x[i],
        top + offset,
        f"{error_percent[i]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=16,
        rotation=45,
    )

plt.xticks(x, names, rotation=45, ha="right", fontsize=16)
plt.ylabel("Yield strength (MPa)", fontsize=18, fontweight="bold")
plt.yticks(fontsize=18)
plt.legend(fontsize=16)
plt.ylim(700, 1.1 * max(actual.max(), predicted.max()) + offset)
plt.tight_layout()
plt.show()


# ============================================================
# Bayesian calibration
# ============================================================

y_surr = y_pred.reshape(-1)
y_exp = y_exp.reshape(-1)

assert y_exp.shape == y_surr.shape, "Experimental and surrogate outputs must align"

# Prior scales (MPa)
BIAS_SD = 50.0
SIGMA_MEAS_SD = 80.0
SIGMA_DISC_SD = 80.0

with pm.Model() as calib_model:
    # Global additive bias (posterior mean tells you if the surrogate is systematically high/low)
    beta0 = pm.Normal("beta0", mu=100.0, sigma=BIAS_SD)

    # Noise components: measurement noise + model discrepancy (lumped, independent)
    sigma_meas = pm.HalfNormal("sigma_meas", sigma=SIGMA_MEAS_SD)
    sigma_disc = pm.HalfNormal("sigma_disc", sigma=SIGMA_DISC_SD)

    # Total predictive uncertainty (not including any extra surrogate error term)
    sigma_total = pm.Deterministic(
        "sigma_total",
        pm.math.sqrt(sigma_meas**2 + sigma_disc**2),
    )

    pm.Normal(
        "y_obs",
        mu=y_surr + beta0,
        sigma=sigma_total,
        observed=y_exp,
    )

    idata = pm.sample(
        draws=1000,
        tune=600,
        chains=4,
        target_accept=0.85,
        cores=2,
        random_seed=RNG_SEED,
        compute_convergence_checks=False,
        progressbar=True,
    )

    idata = pm.sample_posterior_predictive(
        idata,
        extend_inferencedata=True,
        var_names=["y_obs"],
    )


# ----------------------------
# Posterior summary + interpretation notes
# ----------------------------

summary = az.summary(
    idata,
    var_names=["beta0", "sigma_meas", "sigma_disc", "sigma_total"],
    kind="stats",
)
print(summary)

# Interpretation guide (brief, repo-friendly):
# - beta0 (MPa): global additive correction applied to the surrogate mean.
#   * beta0 > 0 suggests the surrogate underpredicts on average (needs upward shift).
#   * beta0 < 0 suggests the surrogate overpredicts on average (needs downward shift).
# - sigma_meas (MPa): estimated measurement noise in the experimental dataset.
# - sigma_disc (MPa): model discrepancy term capturing structural mismatch between surrogate and reality.
# - sigma_total (MPa): combined uncertainty driving the likelihood; use this as the calibrated noise level
#   when propagating uncertainty into inverse design or reliability constraints.
