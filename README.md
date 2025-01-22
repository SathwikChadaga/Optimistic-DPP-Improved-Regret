# Drift Plus Optimistic Penalty

This repository contains simulations of the drift-plus-optimistic-penalty policy from [1]. Please cite [1] if you use any of this repository's code in your work.

[1] S. Chadaga, E. Modiano, "Drift Plus Optimistic Penalty - A Learning Framework for Stochastic Network Optimization," IEEE INFOCOM 2025 - IEEE Conference on Computer Communications, London, 2025.

## Files Description
- `dpop_regret.ipynb` runs the experiment for different values of time horizon and shows the regret curve.
- `regret_sweep.ipynb` shows the regret for different levels of noise and arrival rates.
- `dpop_run.ipynb` runs the experiment for a given time horizon and shows the resulting backlogs and costs.
- `plot_results.ipynb` plots the saved results (use after saving the results from previous programs).
- `utils/` contains utility files required to run the above programs.

## Abstract From Paper
We consider the problem of joint routing and scheduling in queueing networks, where the edge transmission costs are unknown. At each time-slot, the network controller receives noisy observations of transmission costs only for those edges it picks for transmission. The network controller’s objective is to take routing and scheduling decisions so that the total expected cost is minimized. This problem exhibits an exploration-exploitation trade-off, however, previous bandit-style solutions cannot be directly applied to this problem due to the queueing dynamics. In order to ensure network stability, the network controller needs to optimize throughput and cost simultaneously. We show that the best achievable cost is lower bounded by the solution to a static optimization problem, and develop a network control policy using techniques from Lyapunov drift-plus-penalty optimization and multi-arm bandits. We show that the policy achieves a sub-linear regret of order $O(T^{2/3})$, as compared to the best policy that has complete knowledge of arrivals and costs. Finally, we evaluate the proposed policy using simulations and show that its regret is indeed sub-linear.
