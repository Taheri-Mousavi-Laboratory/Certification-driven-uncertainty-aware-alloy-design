"""
global_sensitivity_analysis.py
------------------------------

Train a Polynomial Chaos Expansion (PCE) surrogate model and compute global
Sobol sensitivity indices for alloy composition-property data.

This script is designed as a clean GitHub-ready demo. It uses the same PCE
workflow as the research code:

    1. Load alloy composition and target property data.
    2. Convert input compositions from mass fraction to mole percent.
    3. Define independent uniform distributions for each input element.
    4. Train a PCE surrogate model using Chaospy.
    5. Select the best polynomial order using validation error.
    6. Evaluate the final PCE model on a held-out test set.
    7. Compute first-order and total-effect Sobol sensitivity indices.
    8. Save results and plots.

Notes for users:
    - The example assumes an HDF5 file with:
        "composition" dataset
        "Yield strength (MPa)" dataset

    - The private/raw data file is not included in this repository.
      Replace DATA_PATH with your own local data file.

    - The PCE model is fitted directly in this script and is not loaded
      from a shared .pkl file.

Example:
    python global_sensitivity_analysis.py

Required packages:
    numpy
    pandas
    h5py
    matplotlib
    chaospy
    scikit-learn
"""

import os
import h5py
import numpy as np
import pandas as pd
import chaospy as cp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# =============================================================================
# 1. User settings
# =============================================================================

# Replace this with your own local HDF5 file.
# Do not upload private data files to GitHub.
DATA_PATH = "../Data/Insert_your_file_name.hdf5"

# HDF5 dataset names.
COMPOSITION_KEY = "composition"
TARGET_KEY = "Yield strength (MPa)"

# Output folder for results.
OUTPUT_DIR = "../Results/global_sensitivity"

# Element order in the input composition array.
# The uploaded research code assumes:
#   conc[:, 0] = Ni
#   conc[:, 1] = Cr
#   conc[:, 2] = Nb
#   conc[:, 3] = Mo
#   conc[:, 4] = Ti
#   conc[:, 5] = Al
#   conc[:, 6] = Co
INPUT_ELEMENTS = ["Ni", "Cr", "Nb", "Mo", "Ti", "Al", "Co"]

# Final feature order used for PCE training and plotting.
FEATURE_NAMES = [
    "Ni (mole %)",
    "Cr (mole %)",
    "Nb (mole %)",
    "Al (mole %)",
    "Ti (mole %)",
    "Co (mole %)",
    "Mo (mole %)",
]

# Atomic weights used for mass-fraction to mole-percent conversion.
ATOMIC_WEIGHTS = {
    "Fe": 55.845,
    "Ni": 58.700,
    "Cr": 52.000,
    "Nb": 92.900,
    "Mo": 95.950,
    "Ti": 47.867,
    "Al": 26.98154,
    "Co": 58.933,
}

# PCE/training settings.
RANDOM_SEED = 42
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.1111
POLYNOMIAL_ORDERS = range(1, 5)

# Plot settings.
SAVE_FIGURES = True
SHOW_FIGURES = True


# =============================================================================
# 2. Data loading
# =============================================================================

def load_hdf5_data(data_path, composition_key, target_key):
    """
    Load composition and target property data from an HDF5 file.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 file.
    composition_key : str
        Name of the HDF5 dataset containing compositions.
    target_key : str
        Name of the HDF5 dataset containing target property values.

    Returns
    -------
    compositions : ndarray, shape (n_samples, n_features)
        Input composition array. Expected to be in mass fraction.
    target : ndarray, shape (n_samples,)
        Target property array.
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please update DATA_PATH to point to your local HDF5 file."
        )

    with h5py.File(data_path, "r") as h5_file:
        if composition_key not in h5_file:
            raise KeyError(
                f"Composition dataset '{composition_key}' was not found in {data_path}."
            )

        if target_key not in h5_file:
            raise KeyError(
                f"Target dataset '{target_key}' was not found in {data_path}."
            )

        compositions = np.array(h5_file[composition_key]).astype(np.float64)
        target = np.array(h5_file[target_key]).astype(np.float64)

    if compositions.ndim != 2:
        raise ValueError(
            "The composition dataset must be a 2D array with shape "
            "(n_samples, n_elements)."
        )

    if target.ndim != 1:
        target = target.reshape(-1)

    if len(compositions) != len(target):
        raise ValueError(
            "Composition and target arrays have different lengths:\n"
            f"composition rows = {len(compositions)}, target rows = {len(target)}"
        )

    return compositions, target


# =============================================================================
# 3. Composition conversion
# =============================================================================

def mass_fraction_to_mole_percent(compositions):
    """
    Convert Alloy 718 compositions from mass fraction to mole percent.

    The input composition array is expected to contain:

        [Ni, Cr, Nb, Mo, Ti, Al, Co]

    Fe is treated as the balance element:

        Fe = 1 - sum(Ni, Cr, Nb, Mo, Ti, Al, Co)

    Parameters
    ----------
    compositions : ndarray, shape (n_samples, 7)
        Composition array in mass fraction.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing mole-percent compositions in the feature order:

        Ni, Cr, Nb, Al, Ti, Co, Mo
    """

    if compositions.shape[1] != 7:
        raise ValueError(
            "Expected composition array with 7 columns in this order:\n"
            "[Ni, Cr, Nb, Mo, Ti, Al, Co]"
        )

    mole_percent_data = []

    for row in compositions:
        Ni, Cr, Nb, Mo, Ti, Al, Co = row
        Fe = 1.0 - np.sum(row)

        if Fe < -1e-8:
            raise ValueError(
                "Calculated Fe balance is negative. Check whether your "
                "composition data are in mass fraction instead of wt.%."
            )

        denominator = (
            Fe / ATOMIC_WEIGHTS["Fe"]
            + Ni / ATOMIC_WEIGHTS["Ni"]
            + Cr / ATOMIC_WEIGHTS["Cr"]
            + Nb / ATOMIC_WEIGHTS["Nb"]
            + Mo / ATOMIC_WEIGHTS["Mo"]
            + Ti / ATOMIC_WEIGHTS["Ti"]
            + Al / ATOMIC_WEIGHTS["Al"]
            + Co / ATOMIC_WEIGHTS["Co"]
        )

        Ni_mol = Ni / ATOMIC_WEIGHTS["Ni"] / denominator * 100.0
        Cr_mol = Cr / ATOMIC_WEIGHTS["Cr"] / denominator * 100.0
        Nb_mol = Nb / ATOMIC_WEIGHTS["Nb"] / denominator * 100.0
        Mo_mol = Mo / ATOMIC_WEIGHTS["Mo"] / denominator * 100.0
        Ti_mol = Ti / ATOMIC_WEIGHTS["Ti"] / denominator * 100.0
        Al_mol = Al / ATOMIC_WEIGHTS["Al"] / denominator * 100.0
        Co_mol = Co / ATOMIC_WEIGHTS["Co"] / denominator * 100.0

        mole_percent_data.append([
            Ni_mol,
            Cr_mol,
            Nb_mol,
            Al_mol,
            Ti_mol,
            Co_mol,
            Mo_mol,
        ])

    return pd.DataFrame(mole_percent_data, columns=FEATURE_NAMES)


# =============================================================================
# 4. PCE distribution setup
# =============================================================================

def create_joint_distribution(feature_dataframe):
    """
    Create an independent joint uniform distribution for PCE.

    Each input feature is represented as a uniform distribution between the
    minimum and maximum values observed in the dataset.

    Parameters
    ----------
    feature_dataframe : pandas.DataFrame
        Input feature dataframe.

    Returns
    -------
    joint_distribution : chaospy distribution
        Joint distribution of all input variables.
    """

    distributions = []

    for column in feature_dataframe.columns:
        lower = float(feature_dataframe[column].min())
        upper = float(feature_dataframe[column].max())

        if np.isclose(lower, upper):
            raise ValueError(
                f"Feature '{column}' has identical min and max values. "
                "A uniform distribution cannot be created."
            )

        distributions.append(cp.Uniform(lower, upper))

    joint_distribution = cp.J(*distributions)

    return joint_distribution


# =============================================================================
# 5. PCE training and order selection
# =============================================================================

def split_data(X, y):
    """
    Split the dataset into training, validation, and test sets.

    The split follows the original workflow:
        - 10% test data
        - remaining 90% split into training and validation
        - validation_size = 0.1111 gives approximately 10% of total data

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def select_best_polynomial_order(X_train, y_train, X_val, y_val, joint_distribution):
    """
    Train PCE models with different polynomial orders and select the best order
    based on validation MSE.

    Parameters
    ----------
    X_train : ndarray
        Training input features.
    y_train : ndarray
        Training target values.
    X_val : ndarray
        Validation input features.
    y_val : ndarray
        Validation target values.
    joint_distribution : chaospy distribution
        Joint input distribution.

    Returns
    -------
    best_order : int
        Polynomial order with the lowest validation MSE.
    order_results : pandas.DataFrame
        Validation metrics for each tested order.
    """

    results = []

    for order in POLYNOMIAL_ORDERS:
        polynomial_expansion = cp.expansion.stieltjes(order, joint_distribution)

        pce_model = cp.fit_regression(
            polynomial_expansion,
            X_train.T,
            y_train,
        )

        y_val_pred = pce_model(*X_val.T)

        mse = mean_squared_error(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        results.append({
            "Polynomial order": order,
            "Validation MSE": mse,
            "Validation MAE": mae,
            "Validation R2": r2,
        })

        print(
            f"Order {order}: "
            f"Validation MSE = {mse:.4f}, "
            f"MAE = {mae:.4f}, "
            f"R2 = {r2:.4f}"
        )

    order_results = pd.DataFrame(results)
    best_order = int(
        order_results.loc[
            order_results["Validation MSE"].idxmin(),
            "Polynomial order"
        ]
    )

    print(f"\nBest polynomial order based on validation MSE: {best_order}")

    return best_order, order_results


def train_final_pce_model(X_train, y_train, joint_distribution, polynomial_order):
    """
    Train the final PCE model using the selected polynomial order.

    Parameters
    ----------
    X_train : ndarray
        Training input features.
    y_train : ndarray
        Training target values.
    joint_distribution : chaospy distribution
        Joint input distribution.
    polynomial_order : int
        Selected polynomial order.

    Returns
    -------
    pce_model : chaospy polynomial approximation
        Fitted PCE surrogate model.
    polynomial_expansion : chaospy polynomial expansion
        Polynomial basis used for the PCE model.
    """

    polynomial_expansion = cp.expansion.stieltjes(
        polynomial_order,
        joint_distribution,
    )

    pce_model = cp.fit_regression(
        polynomial_expansion,
        X_train.T,
        y_train,
    )

    return pce_model, polynomial_expansion


# =============================================================================
# 6. Model evaluation
# =============================================================================

def evaluate_model(pce_model, X_test, y_test):
    """
    Evaluate the trained PCE model on a held-out test set.

    Parameters
    ----------
    pce_model : chaospy polynomial approximation
        Fitted PCE surrogate model.
    X_test : ndarray
        Test input features.
    y_test : ndarray
        Test target values.

    Returns
    -------
    y_pred : ndarray
        PCE predictions for the test set.
    metrics : dict
        Test-set evaluation metrics.
    """

    y_pred = pce_model(*X_test.T)

    metrics = {
        "Test MSE": mean_squared_error(y_test, y_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Test MAE": mean_absolute_error(y_test, y_pred),
        "Test R2": r2_score(y_test, y_pred),
    }

    print("\nTest-set performance:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return y_pred, metrics


# =============================================================================
# 7. Sobol sensitivity analysis
# =============================================================================

def compute_sobol_indices(pce_model, joint_distribution, feature_names):
    """
    Compute first-order and total-effect Sobol sensitivity indices from the PCE
    surrogate model.

    Parameters
    ----------
    pce_model : chaospy polynomial approximation
        Fitted PCE surrogate model.
    joint_distribution : chaospy distribution
        Joint input distribution.
    feature_names : list of str
        Names of input features.

    Returns
    -------
    sobol_df : pandas.DataFrame
        DataFrame containing main-effect and total-effect Sobol indices.
    """

    main_effect_indices = cp.Sens_m(pce_model, joint_distribution)
    total_effect_indices = cp.Sens_t(pce_model, joint_distribution)

    sobol_df = pd.DataFrame({
        "Feature": feature_names,
        "Main effect": main_effect_indices,
        "Total effect": total_effect_indices,
    })

    sobol_df["Interaction contribution"] = (
        sobol_df["Total effect"] - sobol_df["Main effect"]
    )

    sobol_df = sobol_df.sort_values(
        by="Total effect",
        ascending=False,
    ).reset_index(drop=True)

    return sobol_df


# =============================================================================
# 8. Plotting functions
# =============================================================================

def plot_feature_distributions(df, target_column, output_dir=None):
    """
    Plot histograms of input features and target property.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing input features and target property.
    target_column : str
        Name of the target-property column.
    output_dir : str or None
        Folder where the plot should be saved.
    """

    columns = list(df.columns)
    n_columns = 3
    n_rows = int(np.ceil(len(columns) / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 3.5 * n_rows),
    )

    axes = axes.flatten()

    for i, column in enumerate(columns):
        axes[i].hist(df[column].dropna(), bins=35, edgecolor="black", alpha=0.75)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Input and target-property distributions", fontsize=14)
    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, "feature_target_distributions.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved distribution plot to: {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def plot_prediction_comparison(y_test, y_pred, output_dir=None):
    """
    Plot predicted vs actual target-property values.

    Parameters
    ----------
    y_test : ndarray
        Actual target-property values.
    y_pred : ndarray
        Predicted target-property values.
    output_dir : str or None
        Folder where the plot should be saved.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_test, y_pred, alpha=0.7, edgecolor="black")

    lower = min(np.min(y_test), np.min(y_pred))
    upper = max(np.max(y_test), np.max(y_pred))

    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black")

    ax.set_xlabel("Actual value")
    ax.set_ylabel("PCE-predicted value")
    ax.set_title("PCE prediction performance")

    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, "pce_prediction_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved prediction comparison plot to: {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def plot_sobol_indices(sobol_df, output_dir=None):
    """
    Plot first-order and total-effect Sobol sensitivity indices.

    Parameters
    ----------
    sobol_df : pandas.DataFrame
        Sobol sensitivity results.
    output_dir : str or None
        Folder where the plot should be saved.
    """

    plot_df = sobol_df.copy()
    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        plot_df["Main effect"],
        width,
        label="Main effect",
        edgecolor="black",
    )

    ax.bar(
        x + width / 2,
        plot_df["Total effect"],
        width,
        label="Total effect",
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Feature"], rotation=45, ha="right")
    ax.set_ylabel("Sobol index")
    ax.set_ylim(0, max(1.0, plot_df["Total effect"].max() * 1.15))
    ax.set_title("Global Sobol sensitivity indices")
    ax.legend()

    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, "sobol_sensitivity_indices.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Sobol plot to: {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


# =============================================================================
# 9. Main workflow
# =============================================================================

def main():
    """
    Run the full global sensitivity analysis workflow.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading data...")
    compositions, target = load_hdf5_data(
        DATA_PATH,
        COMPOSITION_KEY,
        TARGET_KEY,
    )

    print(f"Loaded {len(compositions)} samples.")
    print(f"Composition array shape: {compositions.shape}")
    print(f"Target array shape: {target.shape}")

    print("\nConverting compositions from mass fraction to mole percent...")
    feature_df = mass_fraction_to_mole_percent(compositions)

    full_df = feature_df.copy()
    full_df[TARGET_KEY] = target

    print("\nFeature summary:")
    print(feature_df.describe())

    if SAVE_FIGURES:
        plot_feature_distributions(
            full_df,
            target_column=TARGET_KEY,
            output_dir=OUTPUT_DIR,
        )

    print("\nCreating joint input distribution for PCE...")
    joint_distribution = create_joint_distribution(feature_df)

    X = feature_df.values.astype(np.float64)
    y = target.astype(np.float64)

    print("\nSplitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    print("\nSelecting best polynomial order...")
    best_order, order_results = select_best_polynomial_order(
        X_train,
        y_train,
        X_val,
        y_val,
        joint_distribution,
    )

    order_results_path = os.path.join(OUTPUT_DIR, "pce_order_selection.csv")
    order_results.to_csv(order_results_path, index=False)
    print(f"Saved polynomial-order results to: {order_results_path}")

    print("\nTraining final PCE model...")
    pce_model, polynomial_expansion = train_final_pce_model(
        X_train,
        y_train,
        joint_distribution,
        best_order,
    )

    print("\nEvaluating final PCE model...")
    y_pred, metrics = evaluate_model(
        pce_model,
        X_test,
        y_test,
    )

    metrics_path = os.path.join(OUTPUT_DIR, "pce_test_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved test metrics to: {metrics_path}")

    if SAVE_FIGURES:
        plot_prediction_comparison(
            y_test,
            y_pred,
            output_dir=OUTPUT_DIR,
        )

    print("\nComputing Sobol sensitivity indices...")
    sobol_df = compute_sobol_indices(
        pce_model,
        joint_distribution,
        FEATURE_NAMES,
    )

    sobol_path = os.path.join(OUTPUT_DIR, "sobol_sensitivity_indices.csv")
    sobol_df.to_csv(sobol_path, index=False)

    print("\nSobol sensitivity results:")
    print(sobol_df)

    print(f"\nSaved Sobol sensitivity results to: {sobol_path}")

    if SAVE_FIGURES:
        plot_sobol_indices(
            sobol_df,
            output_dir=OUTPUT_DIR,
        )

    print("\nGlobal sensitivity analysis completed.")


if __name__ == "__main__":
    main()
