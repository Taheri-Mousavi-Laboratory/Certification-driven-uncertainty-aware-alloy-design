# Source Code

This folder contains the main Python source files for uncertainty-aware Alloy 718 design.

## Folders

### `Inverse design`

Contains scripts for Bayesian optimization-based inverse design.

These scripts can be used to search for Alloy 718 composition windows that satisfy target strength, uncertainty, and reliability requirements.

### `Local_sensitivity_analysis`

Contains scripts or notebooks for local sensitivity analysis.

These files can be used to evaluate how small changes in alloy composition affect the predicted yield strength.

### `Model training and UQ`

Contains scripts for surrogate model training, model validation, uncertainty quantification, and Bayesian calibration.

These files can be used to train the PCE surrogate model, validate predictions, and estimate model uncertainty.

## Main tasks

The code in this folder can be used to:

1. Train a surrogate model for Alloy 718 yield strength prediction.
2. Validate model predictions against reference or experimental data.
3. Quantify model uncertainty using Bayesian calibration.
4. Perform local and global sensitivity analysis.
5. Run inverse design to identify robust composition windows.
