# Demo Notebooks

This folder contains three demo notebooks for Alloy 718. Two use a trained Polynomial Chaos Expansion (PCE) surrogate model for inverse design and sensitivity analysis, and one demonstrates the physics-based yield strength model.

The notebooks show how the surrogate model can be used to identify robust alloy composition windows and evaluate the local effect of compositional changes on predicted yield strength.

## Notebooks

### Demo_inverse_design.ipynb

This notebook demonstrates uncertainty-aware inverse design of Alloy 718 composition windows using Bayesian optimization.

It can be used to:

* Search for composition windows that satisfy a target yield strength
* Include uncertainty limits in the design process
* Include reliability or coverage requirements
* Compare full and reduced design spaces
* Identify filtered candidate composition windows
* Find Pareto-optimal design candidates
* Visualize selected design windows

The notebook includes two workflows.

#### Seven-element workflow

In this workflow, Bayesian optimization controls all seven alloying elements:

`Ni, Cr, Nb, Al, Ti, Co, Mo`

This workflow is useful for searching the full Alloy 718 composition space.

#### Three-element workflow

In this workflow, Bayesian optimization controls only:

`Nb, Al, Ti`

The remaining elements are allowed to vary within fixed Alloy 718 composition ranges:

`Ni, Cr, Co, Mo`

This reduced workflow is useful for testing whether robust design can be achieved by controlling only the most influential elements.

### Demo_local_sensitivity_analysis.ipynb

This notebook demonstrates local sensitivity analysis using the trained PCE surrogate.

It can be used to:

* Enter a base Alloy 718 composition
* Calculate local sensitivity for each element
* Estimate the strength change caused by small composition perturbations
* Rank elements by their local influence
* Visualize normalized sensitivity values

The local sensitivity is calculated using the analytical derivative of the PCE surrogate:

`dY/dx`

where `Y` is the predicted yield strength and `x` is the concentration of each element.

## Required Files

The notebooks require a trained PCE surrogate model and an example dataset. These files are not included in this demo folder.

Update the paths inside the notebooks as needed:

`SURROGATE_MODEL_PATH = Path("../model/pce_surrogate.pkl")`

The PCE surrogate model expects compositions in wt.% in the following order:

`Ni, Cr, Nb, Al, Ti, Co, Mo`

## How to Run

Open the notebooks using Jupyter Notebook, JupyterLab, or VS Code.

Run the cells from top to bottom.

The notebooks are interactive. Depending on the workflow, they may ask for:

* Target yield strength
* Allowable uncertainty threshold
* Reliability target
* Objective weights
* Number of Bayesian optimization iterations
* Batch size
* Minimum composition-window width
* Base composition
* Composition perturbations

## Outputs

The inverse-design notebook may generate files such as:

* `bo_measurements_full.csv`
* `filtered_candidates.csv`
* `pareto_front.csv`
* `coordinate_plot.png`

The local sensitivity notebook may generate files such as:

* `local_sensitivity_results.csv`
* `local_sensitivity_plot.png`

## Demo_Yield Strength Model.ipynb

A demo notebook, `Demo_yield_strength_model.ipynb`, has been added to show how to run the physics-based yield strength model. It does not use the PCE surrogate.

### Inputs

* CALPHAD-derived phase data for each composition:
  * Composition (mass fractions of `Ni, Cr, Nb, Mo, Ti, Al, Co`, with Fe as balance)
  * γ′, γ″ and γ matrix phase fractions
  * γ′, γ″ and matrix lattice parameters
  * Solid-solution strength
* A fixed grain size (µm) for the Hall–Petch contribution

If no phase data is provided, randomly generated placeholder data is used to demonstrate the workflow.

### Outputs

* A table (`yield_strength_results.csv`) with, for each composition:
  * Hall–Petch, solid-solution, γ′ (Orowan) and γ″ (coherency) strength contributions
  * Total yield strength
