# Experiment Overview: Potential-Based Reward Shaping (PBRS) on Acrobot-v1

## Research Motivation
Provide a dense training signal to accelerate early learning while mathematically guaranteeing policy invariance (the optimal policy of the shaped MDP remains identical to the original MDP).

## Research Hypothesis
* **Hypothesis**: PBRS will improve early sample efficiency (e.g. time to reach reward thresholds of 100, 200, 300) compared to Identity, and avoid any policy subversion risks associated with heuristic dense rewards, converging to the exact same optimal policy.

## Reward Function Mathematical Formulation
The shaped reward is defined as:
$$R_{shaped}(s, a, s') = R_{original}(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

## Implementation Details
* **Class**: `PbrsRewardShaper` inside `reward_functions/pbrs.py`
* **Mechanics**: Computes differences in the state potential function $\Phi(s) = - (w_x |x| + w_v |v| + w_\theta |\theta| + w_\omega |\omega|)$, with special boundary condition $\Phi(s_{\text{terminal}}) = 0$.

## Configuration Details
Hyperparameters and parameters are archived under:
* [config.yaml](raw/config.yaml)
* Environment: `Acrobot-v1`
* Hyperparameters: Standard SB3 PPO default MLP policy network (`[64, 64]`) on CPU.

## Observations
- Convergence: The agent converges stably to the ceiling performance (500 steps).
- Optimality: Achieves maximum task success rate.
- Limitations: Slower convergence rate in early phases compared to potential dense functions.
- Next Actions: Benchmark against Potential-Based Reward Shaping (PBRS) to improve early convergence while preserving policy constraints.
