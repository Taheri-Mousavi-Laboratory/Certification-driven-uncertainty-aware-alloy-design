Global sensitivity analysis

The global_sensitivity_analysis.py script trains a Polynomial Chaos Expansion
(PCE) surrogate model using Chaospy and computes Sobol sensitivity indices for
each alloying element. The raw HDF5 data file used in the study is not included
in this repository. Users should replace DATA_PATH with the location of their
own HDF5 file.

Expected HDF5 datasets:
    composition
    Yield strength (MPa)

The composition dataset should contain mass fractions in the following order:
    [Ni, Cr, Nb, Mo, Ti, Al, Co]

The script converts mass fractions to mole percent, trains the PCE model,
selects the best polynomial order using validation error, evaluates the model
on a test set, and saves Sobol sensitivity results as a CSV file.