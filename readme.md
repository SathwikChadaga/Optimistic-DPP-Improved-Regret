# Drift Plus Optimistic Penalty - Improved $O(\sqrt{T})$ Regret

This repository contains simulations of the drift-plus-optimistic-penalty policy from [1] and [2]. 

Please cite this work using [1] if you use any of this repository's code in your work.

[1] S. Chadaga and E. Modiano, "Drift Plus Optimistic Penalty – A Learning Framework for Stochastic Network Optimization," IEEE INFOCOM 2025 - IEEE Conference on Computer Communications, London, United Kingdom, 2025, pp. 1-10, doi: 10.1109/INFOCOM55648.2025.11044621. 

[2] S. Chadaga and E. Modiano, "Drift Plus Optimistic Penalty: A Learning Framework for Stochastic Network Optimization with Improved Regret Bounds," arXiv.org, Sep. 03, 2025. https://arxiv.org/abs/2509.03762.

## Files Description
- `dpop_regret.ipynb` runs the experiment for different values of time horizon and shows the regret curve.
- `regret_sweep.ipynb` shows the regret for different levels of noise and arrival rates.
- `dpop_run.ipynb` runs the experiment for a given time horizon and shows the resulting backlogs and costs.
- `plot_results.ipynb` plots the saved results (use after saving the results from previous programs).
- `utils/` contains utility files required to run the above programs.

## Abstract From Paper
We consider the problem of joint routing and scheduling in queueing networks, where the edge transmission costs are unknown. At each time-slot, the network controller receives noisy observations of transmission costs only for those edges it selects for transmission. The network controller's objective is to make routing and scheduling decisions so that the total expected cost is minimized. This problem exhibits an exploration-exploitation trade-off, however, previous bandit-style solutions cannot be directly applied to this problem due to the queueing dynamics. In order to ensure network stability, the network controller needs to optimize throughput and cost simultaneously. We show that the best achievable cost is lower bounded by the solution to a static optimization problem, and develop a network control policy using techniques from Lyapunov drift-plus-penalty optimization and multi-arm bandits. We show that the policy achieves a sub-linear regret of order $O(\sqrt{T}\log T)$, as compared to the best policy that has complete knowledge of arrivals and costs. Finally, we evaluate the proposed policy using simulations and show that its regret is indeed sub-linear.
