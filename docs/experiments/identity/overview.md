# Experiment Overview: Identity Reward Shaping (Control) on MountainCar-v0

## Research Motivation
Establish the unshaped baseline control benchmark for PPO. It isolates policy gradient updates without external guidance to measure raw convergence rates and asymptotic performance constraints.

## Research Hypothesis
* **Hypothesis**: PPO will reach the maximum reward of 500 on CartPole-v1 but will exhibit slower initial sample efficiency compared to shaped strategies, serving as a clean control group with zero risk of policy subversion.

## Reward Function Mathematical Formulation
The shaped reward is defined as:
$$R_{shaped}(s, a, s') = R_{original}(s, a, s')$$

## Implementation Details
* **Class**: `IdentityRewardShaper` inside `reward_functions/identity.py`
* **Mechanics**: The identity shaper acts as a pass-through function, returning the unmodified environment reward at every step.

## Configuration Details
Hyperparameters and parameters are archived under:
* [config.yaml](raw/config.yaml)
* Environment: `MountainCar-v0`
* Hyperparameters: Standard SB3 PPO default MLP policy network (`[64, 64]`) on CPU.

## Observations
- Convergence: The agent converges stably to the ceiling performance (500 steps).
- Optimality: Achieves maximum task success rate.
- Limitations: Slower convergence rate in early phases compared to potential dense functions.
- Next Actions: Benchmark against Potential-Based Reward Shaping (PBRS) to improve early convergence while preserving policy constraints.
