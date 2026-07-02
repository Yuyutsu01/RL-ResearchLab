# Experimental Methodology

To compare different reward shaping strategies scientifically, we must ensure that all other experimental variables remain constant. This document outlines our experimental methodology.

---

## 1. Core Design Principle: Decoupling

We enforce a strict decoupling of:
1. **Environment Physics**: The physical dynamics, observation spaces, and base reward outputs (e.g. standard `CartPole-v1`).
2. **Reward Modification**: The shaping math ($F(s, a, s')$), isolated in the `RewardShaper` interface.
3. **Agent Optimization**: The PPO model implementation in Stable-Baselines3.

```
+---------------------------+
|    Raw Gymnasium Env      |
+---------------------------+
              | (obs, reward, done, info)
              v
+---------------------------+
|   RewardShapingWrapper    | <--- [ RewardShaper Interface ]
+---------------------------+
              | (obs, shaped_reward, done, info)
              v
+---------------------------+
|    Stable-Baselines3      |
|        PPO Agent          |
+---------------------------+
```

PPO interacts only with the wrapped environment and remains unaware that rewards are modified. This guarantees that we can plug in different shaping methods without altering the underlying learning algorithm.

---

## 2. Separate Training and Evaluation Environments

Evaluating an agent on the same reward function it is optimizing can introduce biases. For example, if a shaped reward is denser and yields higher numerical sums, the agent's training curves will look artificially inflated, even if it has not learned to solve the original task.

To address this, our methodology enforces a strict split:
* **Training Environment**: Wrapped with the target `RewardShaper` (e.g., `IdentityRewardShaper`, `DenseRewardShaper`). PPO optimizes the shaped reward.
* **Evaluation Environment**: Wrapped with the `IdentityRewardShaper` (raw reward). Periodically (e.g., every 5,000 steps), the policy is evaluated deterministically for 10 episodes in this unshaped environment to measure its true performance on the baseline physics objective.

This guarantees that all strategies are compared fairly against the exact same unshaped objective.

---

## 3. Data Alignment and Aggregation

Because reinforcement learning episodes terminate at variable steps, different seeds will output monitor entries at different step counts. To aggregate them:
1. **Grid Interpolation**: We define a common steps grid of $K = 100$ points spanning from $0$ to the maximum step limit (e.g. 100,000).
2. **Averaging**: We smooth individual runs using a rolling window, then interpolate the metrics for each seed onto the common steps grid.
3. **Statistics**: We calculate the cross-seed mean, median, standard deviation, and 95% confidence intervals at each step grid point.
