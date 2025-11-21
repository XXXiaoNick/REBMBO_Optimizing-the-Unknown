# REBMBO: Optimizing the Unknown with GP + EBM + RL

This repository contains a reference implementation of **REBMBO** (Reinforced Energy-Based Model for Bayesian Optimization), a black-box optimization framework that combines:

- **Gaussian Processes (GPs)** for accurate *local* modeling,
- **Energy-Based Models (EBMs)** for *global* structure and exploration,
- **Proximal Policy Optimization (PPO)** for *multi-step* planning in Bayesian optimization.

REBMBO is designed for expensive black-box objectives where gradients are unavailable and evaluations are costly (e.g., hyperparameter tuning, nanophotonic design, protein engineering).

---

## 1. Method Overview

### 1.1 High-level idea

At a high level, REBMBO does the following in each BO iteration:

1. **Update a GP surrogate** to get local posterior mean and uncertainty.
2. **Train an EBM** via short-run MCMC to learn a global “energy landscape” over the input space.
3. **Form a state** that concatenates GP statistics and EBM energy signals.
4. **Use PPO** to select query points via multi-step planning, balancing:
   - high function values, and  
   - low EBM energy (globally promising basins).

This loop continues under a fixed evaluation budget, returning the best point found.

### 1.2 Three modules (A / B / C)

- **Module A – GP surrogate (local modeling)**  
  - Supports three variants:
    - `REBMBO-C`: Classic GP with exact O(n³) inference (for moderate n, low–mid dimensions).
    - `REBMBO-S`: Sparse GP with inducing points (for larger n / higher dimensions).
    - `REBMBO-D`: Deep kernel GP (neural feature extractor + GP) for multi-scale, non-stationary structure.
  - GPs provide posterior mean μₜ(x) and std σₜ(x) used by acquisition and RL.

- **Module B – EBM-driven global exploration**  
  - EBM defines unnormalized density:  
    \\( p_\theta(x) \propto \exp(-E_\theta(x)) \\).
  - Trained by **short-run MCMC-based MLE**:
    - Positive phase: lower energy on data points.
    - Negative phase: raise energy on model-generated samples (Langevin / SGLD).
  - After training, we plug energy into a UCB-style acquisition:

    \\[
    \alpha_{\text{EBM-UCB}}(x) = \mu_{f,t}(x) + \beta \sigma_{f,t}(x) - \gamma E_\theta(x),
    \\]

    where:
    - β controls exploration via GP uncertainty,
    - γ controls how strongly EBM guides us toward low-energy (globally promising) basins.

- **Module C – PPO-based multi-step planning**  
  - Treat each BO iteration as an MDP:
    - **State**:  
      \\( s_t = (\mu_{f,t}(x), \sigma_{f,t}(x), E_\theta(x)) \\)
    - **Action**: next query point \\( a_t \in \mathcal{X} \\).
    - **Reward**:

      \\[
      r_t(s_t, a_t) = n f(a_t) - \lambda E_\theta(a_t),
      \\]

      where λ controls the trade-off between immediate function value and global exploration.
  - PPO optimizes a stochastic policy π_ϕ(a | s) using the standard clipped objective to avoid large, unstable updates.
  - This transforms single-step acquisition into **multi-step lookahead**.

---

## 2. Landscape-Aware Regret (LAR)

Standard regret only measures how far f(xₜ) is from the global optimum f(x\*). REBMBO introduces **Landscape-Aware Regret (LAR)** to incorporate both local and global performance:

\\[
R_t^{\text{LAR}} =
\big(f(x^\*) - f(x_t)\big)
+ \alpha \big( E_\theta(x^\*) - E_\theta(x_t) \big),
\\]

- α ≥ 0 balances local optimality vs. global coverage.
- For non-energy-based baselines, α = 0 (standard regret).
- For REBMBO with EBM, α > 0 yields a more holistic metric.

Both LAR and standard regret / normalized objective are reported in experiments.

---

## 3. Repository Structure (suggested)

```text
.
├── README.md                  # This file
├── src/
│   ├── rebmbo/
│   │   ├── gp_module.py       # Module A: GP variants (C, S, D)
│   │   ├── ebm_module.py      # Module B: EBM + short-run MCMC
│   │   ├── ppo_module.py      # Module C: PPO agent + MDP interface
│   │   ├── acquisition.py     # EBM-UCB implementation
│   │   └── algorithm.py       # REBMBO training loop (Algorithm 1)
│   ├── baselines/             # TuRBO, BALLET-ICI, EARL-BO, 2-step EI, KG, etc.
│   └── utils/                 # logging, metrics, plotting, configs
├── configs/
│   ├── branin_2d.yaml
│   ├── ackley_5d.yaml
│   ├── rosenbrock_8d.yaml
│   ├── hdbo_200d.yaml
│   ├── nanophotonic_3d.yaml
│   ├── rosetta_86d.yaml
│   ├── natsbench_20d.yaml
│   └── robot_traj_40d.yaml
├── experiments/
│   ├── run_synthetic.sh       # scripts to reproduce synthetic tasks
│   └── run_real_world.sh      # scripts to reproduce real benchmarks
├── data/                      # task-specific datasets / interfaces
└── results/
    ├── logs/
    ├── figures/
    └── tables/
