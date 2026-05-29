# Certifcation-driven uncertainty-aware alloy design

> A machine learning framework for **uncertainty-quantified forward prediction** and **inverse design** of alloy compositions — demonstrated on Ni-based superalloys (Alloy 718).

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

---

## Overview

This repository provides a modular, end-to-end pipeline for data-driven alloy design. Starting from a composition–property dataset, you can:

1. **Train a surrogate ML model** and validate it with Bayesian calibration
2. **Quantify prediction uncertainty** via ensemble methods
3. **Analyze sensitivity** — identify which alloying elements most influence properties (locally and globally)
4. **Run inverse design** — given a target property range, automatically suggest candidate elemental compositions

The framework is designed to be **adaptable**: researchers can substitute their own alloy system, target property, and dataset.

---

## Repository Structure

```
Uncertainty-aware-inverse-design-of-alloys/
│
├── Data/                                    # Data directory (dataset not included — see note below)
│
├── Demo/                                    # Runnable demo notebooks
│   ├── Demo for Local sensitivity analysis  # Interactive local sensitivity walkthrough
│   ├── Demo_inverse_design.ipynb            # End-to-end inverse design demo
│   └── readme_demo.md                       # Demo usage instructions
│
├── Model/                                   # Saved trained model files (generated after training)
│
├── Python_scripts/
│   │
│   ├── Model training and UQ/               # Step 1: Train model + quantify uncertainty
│   │   ├── Model_training_with_Global_sens  # Train surrogate + compute global sensitivity
│   │   ├── Model validation and Bayesian ca # Model validation + Bayesian uncertainty calibration
│   │   └── readme.md
│   │
│   ├── Local_sensitivity_analysis/          # Step 2: Local sensitivity at a given composition
│   │   ├── Local sensitivity analysis_OAT   # One-at-a-time (OAT) perturbation method
│   │   ├── Local_sensitivity_analysis_deriv # Derivative-based sensitivity method
│   │   ├── readme_Local_SA.md
│   │   └── readme_Global_SA_.md
│   │
│   └── Inverse design/                      # Step 3: Inverse composition search
│       ├── Inverse_design_of_3_elements_B   # Inverse design for 3-element alloy system
│       ├── Inverse_design_of_7_elements_B   # Inverse design for 7-element alloy system
│       └── readme.md
│
├── ML_env.yml                               # Conda environment specification
└── README.md
```
## Getting Started

### 1. Clone the repository

bash
git clone https://github.com/Hasan-024/Uncertainty-aware-inverse-design-of-alloys.git
cd Uncertainty-aware-inverse-design-of-alloys


### 2. Set up the environment

bash
conda env create -f ML_env.yml
conda activate ML_env


### 3. Prepare your data

Place your dataset in the `Data/` directory. The pipeline expects a CSV or Excel file with:
- **Feature columns**: elemental compositions in wt.% (e.g., Ni, Cr, Fe, Mo, Nb, Ti, Al) and optional processing parameters
- **Target column**: the mechanical property to predict (e.g., yield strength in MPa)

> **Note**: The original training dataset is not publicly distributed. A synthetic sample dataset is included in `Data/` for testing purposes.

---

## Workflow

The pipeline follows three sequential steps. Each has its own subfolder under `Python_scripts/` with a dedicated `readme.md`.

---

### Step 1 — Model Training & Uncertainty Quantification

**Folder**: `Python_scripts/Model training and UQ/`

| Script | What it does |
|---|---|
| `Model_training_with_Global_sens` | Trains a surrogate ML model on your composition–property data and computes global sensitivity indices (e.g., Sobol indices via `chaospy`) |
| `Model validation and Bayesian ca` | Validates model performance and applies Bayesian calibration to produce well-calibrated uncertainty estimates |

**Run model training:**
```bash
# See Python_scripts/Model training and UQ/readme.md for full usage
```

**Outputs saved to** `Model/`: trained model file, scaler, feature metadata.

---

### Step 2 — Sensitivity Analysis

**Folder**: `Python_scripts/Local_sensitivity_analysis/`

Two complementary methods are provided. Use either or both:

| Script | Method | When to use |
|---|---|---|
| `Local sensitivity analysis_OAT` | One-at-a-time (OAT) perturbation | Fast; perturbs each element ±δ and records the change in predicted property |
| `Local_sensitivity_analysis_deriv` | Derivative-based (finite-difference) | More precise gradient estimate; better for smooth surrogate models |

Both scripts take a **base composition** as input and output an elemental sensitivity ranking — telling you which elements most strongly influence the predicted property *at that specific composition point*.

**Example usage:**
```bash
# Provide a base composition in wt.% — see readme_Local_SA.md for full options
```

> For **global** sensitivity (element importance averaged over the full composition space), refer to `readme_Global_SA_.md` — global indices are computed as part of Step 1.

---

### Step 3 — Inverse Design

**Folder**: `Python_scripts/Inverse design/`

Given a target property range, these scripts search for candidate alloy compositions predicted to satisfy it, ranked by uncertainty.

| Script | Alloy system |
|---|---|
| `Inverse_design_of_3_elements_B` | 3-element system (reduced composition space) |
| `Inverse_design_of_7_elements_B` | 7-element system (full Alloy 718 composition space) |

Choose the script that matches your design problem. Both accept a target property range and optional per-element bounds as inputs.

**Example usage:**
```bash
# See Python_scripts/Inverse design/readme.md for full usage and parameter options
```

**Outputs**: A ranked table of candidate compositions with predicted mean and uncertainty, plus a plot of the prediction vs. uncertainty trade-off.

---

## Demo Notebooks

Interactive Jupyter notebooks for quick exploration without modifying scripts:

| Notebook | What it covers |
|---|---|
| `Demo/Demo for Local sensitivity analysis` | Walk through local sensitivity analysis interactively on a sample composition |
| `Demo/Demo_inverse_design.ipynb` | Full inverse design pipeline — from loading a model to generating ranked candidates |

Start here if you want to **explore the framework** before adapting it to your own data.

---

## Standalone Demo (Windows, No Python Required)

A Windows executable is available for users who want to explore **local sensitivity analysis** without installing Python.

**Download**: [Releases page](https://github.com/Hasan-024/Uncertainty-aware-inverse-design-of-alloys/releases) → `Alloy718_Local_Sensitivity_Demo.exe`

1. Double-click the executable
2. Enter a base alloy composition (wt.%)
3. Enter expected variation ranges per element
4. View predicted yield strength, uncertainty bounds, and elemental sensitivity scores

All model files are bundled — no installation needed.

---

## Dependencies

All dependencies are managed via `ML_env.yml`. Key packages:

| Package | Purpose |
|---|---|
| `scikit-learn` | Surrogate model training and evaluation |
| `numpy` / `pandas` | Numerical computing and data handling |
| `chaospy` | Surrogate modeling and global sensitivity analysis (Sobol indices) |
| `matplotlib` / `seaborn` | Visualization |
| `scipy` | Statistical utilities |

---

## Contact

For questions or collaboration, email: hjame@andrew.cmu.edu
