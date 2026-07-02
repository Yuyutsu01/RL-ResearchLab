# Experiment Overview: Dense Reward Shaping on CartPole-v1

## Research Motivation
Provide a continuous, immediate gradient signal at every transition step to guide the agent toward the track center and upright balance posture, improving early-stage sample efficiency.

## Research Hypothesis
* **Hypothesis**: The dense shaper will lead to faster early-stage convergence (requiring fewer steps to reach intermediate rewards like 200, 300) compared to the unshaped baseline. However, there is a risk of policy subversion or boundary instability if weights are not properly balanced.

## Reward Function Mathematical Formulation
The shaped reward is defined as:
$$R_{shaped}(s, a, s') = R_{original}(s, a, s') + \text{max\_bonus} - (\text{position\_weight} \cdot |x'| + \text{angle\_weight} \cdot |\theta'|)$$

## Implementation Details
* **Class**: `DenseRewardShaper` inside `reward_functions/dense.py`
* **Mechanics**: Calculates linear penalties on the next state's cart position displacement $|x'|$ and pole angular tilt $|	heta'|$, subtracting them from the base reward.

## Configuration Details
Hyperparameters and parameters are archived under:
* [config.yaml](raw/config.yaml)
* Environment: `CartPole-v1`
* Hyperparameters: Standard SB3 PPO default MLP policy network (`[64, 64]`) on CPU.

## Observations
- Convergence: The agent converges stably to the ceiling performance (500 steps).
- Optimality: Achieves maximum task success rate.
- Limitations: Slower convergence rate in early phases compared to potential dense functions.
- Next Actions: Benchmark against Potential-Based Reward Shaping (PBRS) to improve early convergence while preserving policy constraints.
