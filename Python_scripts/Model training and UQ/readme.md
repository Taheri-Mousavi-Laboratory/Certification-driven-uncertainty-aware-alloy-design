# Model Training, Sensitivity Analysis, and Bayesian Calibration Scripts

This folder contains Python scripts for training and evaluating surrogate models for Alloy 718 property prediction.

## Files

### `Model_training_with_Global_sensitivity_analysis.py`

Trains a Polynomial Chaos Expansion (PCE) surrogate model using alloy composition-property data.

This script can be used to:

- load composition-property data,
- train a PCE surrogate model,
- evaluate model prediction performance,
- calculate global Sobol sensitivity indices,
- identify which elements have the strongest global effect on the predicted property.

### `Model validation and Bayesian calibration.py`

Validates model predictions against reference or experimental data and performs Bayesian calibration.

This script can be used to:

- compare predicted and reported property values,
- evaluate model accuracy,
- estimate model uncertainty,
- calibrate uncertainty parameters,
- provide uncertainty values that can be used in inverse design workflows.

## How to Run

Run the scripts from a terminal or IDE:

python "Model_training_with_Global_sensitivity_analysis.py"

## Outputs

Depending on the script, outputs may include:

trained surrogate model files,
prediction accuracy results,
validation plots,
Bayesian calibration results,
global sensitivity indices,
sensitivity plots.
