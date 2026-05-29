Local sensitivity analysis

The local_sensitivity_analysis.py script estimates the local effect of small
composition changes on predicted yield strength using a trained PCE surrogate.
The script computes analytical derivatives of the PCE polynomial with respect
to each alloying element and reports dY/dx in MPa per wt.%.

The trained PCE model file is not included in this repository. Users should
provide their own local model path using:

    --model_path ../Model/PCE_surrogate_model.pkl

The expected element order is:

    [Ni, Cr, Nb, Al, Ti, Co, Mo]

Example usage:

    python local_sensitivity_analysis.py \
        --model_path ../Model/PCE_surrogate_model.pkl \
        --base_composition "Ni:52,Cr:19,Nb:5,Al:0.5,Ti:1.0,Co:0.5,Mo:3.2" \
        --delta_composition "Ni:0.1,Cr:0.1,Nb:0.05,Al:0.05,Ti:-0.05,Co:0.05,Mo:0.1"

The output includes:
    local_sensitivity_results.csv
    local_sensitivity_ranking.png
    local_strength_change.png