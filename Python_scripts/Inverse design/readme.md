# Inverse Design Scripts

This folder contains two Python scripts for uncertainty-aware inverse design of Alloy 718 using a trained PCE surrogate model.

## Files

### `inverse_design_7element.py`

Optimizes composition windows for all seven elements:

`Ni, Cr, Nb, Al, Ti, Co, Mo`

### `inverse_design_3element.py`

Optimizes only three elements:

`Nb, Al, Ti`

The remaining elements, `Ni, Cr, Co, Mo`, are sampled within fixed Alloy 718 ranges.

Ouputs are saved as:

bo_measurements_full.csv
filtered_candidates.csv
pareto_front.csv
coordinate_plot.png
