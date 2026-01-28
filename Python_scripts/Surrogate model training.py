"""
Train a Polynomial Chaos Expansion (PCE) surrogate model (ChaosPy) to predict
Alloy 718 yield strength from 7 composition features (wt%).

Workflow:
  1) Load composition–strength dataset from CSV
  2) Build a joint Uniform input distribution using observed min/max ranges
  3) Split data into train/validation/test sets
  4) Select polynomial order via validation MSE
  5) Train final PCE model and report test metrics
"""


import numpy as np
import pandas as pd
import chaospy as cp
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


# ----------------------------
# Configuration
# ----------------------------
RNG_SEED = 42
DATA_PATH = Path("data/Alloy718_comp_to_strength.csv")

FEATURE_COLS = [
    "Ni (wt%)",
    "Cr (wt%)",
    "Nb (wt%)",
    "Al (wt%)",
    "Ti (wt%)",
    "Co (wt%)",
    "Mo (wt%)",
]
TARGET_COL = "Yield strength (MPa)"

TEST_SIZE = 0.10
VAL_FRACTION_OF_TRAINVAL = 0.1111  # ~10% of total when trainval is 90%
POLY_ORDERS = range(1, 4)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    return df


def build_joint_distribution(df: pd.DataFrame) -> cp.Distribution:
    """Create a joint Uniform distribution over observed feature ranges."""
    marginals = []
    for col in FEATURE_COLS:
        lo = float(df[col].min())
        hi = float(df[col].max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid range for {col}: min={lo}, max={hi}")
        marginals.append(cp.Uniform(lo, hi))
    return cp.J(*marginals)


def main() -> None:
    np.random.seed(RNG_SEED)

    # --- Load data ---
    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)

    # --- Define distributions from observed ranges ---
    joint_dist = build_joint_distribution(df)

    # --- Train-test split ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RNG_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VAL_FRACTION_OF_TRAINVAL,
        random_state=RNG_SEED,
    )

    # --- Polynomial order search ---
    val_mse = []
    for order in POLY_ORDERS:
        poly_expansion = cp.expansion.stieltjes(order, joint_dist)
        approx_model = cp.fit_regression(poly_expansion, X_train.T, y_train)

        y_pred_val = approx_model(*X_val.T)
        mse = mean_squared_error(y_val, y_pred_val)
        mape = mean_absolute_percentage_error(y_val, y_pred_val)

        val_mse.append(mse)
        print(f"Order {order}: Validation MSE = {mse:.4f}, MAPE = {mape:.4f}")

    best_order = list(POLY_ORDERS)[int(np.argmin(val_mse))]
    print(f"Best polynomial order: {best_order}")

    # --- Train final PCE model with best order ---
    poly_expansion = cp.expansion.stieltjes(best_order, joint_dist)
    approx_model = cp.fit_regression(poly_expansion, X_train.T, y_train)

    # --- Evaluate deterministic performance ---
    y_pred_test = approx_model(*X_test.T)
    mse = mean_squared_error(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print(f"Test MSE = {mse:.4f}, MAPE = {mape:.4f}, R² = {r2:.4f}")


if __name__ == "__main__":
    main()
