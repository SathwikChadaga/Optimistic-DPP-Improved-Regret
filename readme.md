# Drift Plus Optimistic Penalty - Improved $O(\sqrt{T})$ Regret

This repository contains simulations of the drift-plus-optimistic-penalty policy from [1]. Please cite [1] if you use any of this repository's code in your work.

[1] S. Chadaga and E. Modiano, "Drift Plus Optimistic Penalty – A Learning Framework for Stochastic Network Optimization," IEEE INFOCOM 2025 - IEEE Conference on Computer Communications, London, United Kingdom, 2025, pp. 1-10, doi: 10.1109/INFOCOM55648.2025.11044621. 

## Files Description
- `dpop_regret.ipynb` runs the experiment for different values of time horizon and shows the regret curve.
- `regret_sweep.ipynb` shows the regret for different levels of noise and arrival rates.
- `dpop_run.ipynb` runs the experiment for a given time horizon and shows the resulting backlogs and costs.
- `plot_results.ipynb` plots the saved results (use after saving the results from previous programs).
- `utils/` contains utility files required to run the above programs.

## Abstract From Paper

