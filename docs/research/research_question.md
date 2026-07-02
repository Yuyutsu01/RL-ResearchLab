# Core Research Question

The primary research question of this study is:

> **How do different reward shaping strategies affect the convergence speed, sample efficiency, training stability, cumulative reward, and policy robustness of Proximal Policy Optimization (PPO) agents across different reinforcement learning environments?**

---

## Context and Significance

Reinforcement learning (RL) agents learn through trial and error, guided by a scalar reward function. In classic control tasks (and many real-world environments), this reward function is often sparse or coarse, giving the agent very little guidance until a task is fully completed or failed. While theoretically correct, sparse rewards lead to extreme sample inefficiency.

**Reward Shaping** alters the reward signal to guide agent exploration. However, modifying the reward function can introduce several complications:
1. **Policy Subversion**: The agent may learn to maximize the shaped reward in unintended ways, leading to behaviors that do not solve the original problem (e.g., a boat spinning in circles to collect coins instead of completing a race).
2. **Optimization Dynamics**: Even when policies are preserved (e.g., using Potential-Based Reward Shaping), the altered optimization landscape can change gradient steps, affecting training stability, policy entropy, and convergence rates.
3. **Robustness and Sensitivity**: Shaping parameters (e.g., scaling factors, distance definitions) add new hyperparameters. If the agent's performance is highly sensitive to these parameters, the strategy may lack robustness.

---

## Specific Inquiries

We decompose our primary research question into the following sub-inquiries:

1. **Convergence Speed**: How many environment interaction steps does PPO require to reach a pre-defined performance threshold (e.g., a rolling average score of 475 out of 500 on `CartPole-v1`) under each shaping strategy?
2. **Sample Efficiency**: What is the overall learning rate, and how quickly does the policy gradient step toward the target objective?
3. **Training Stability**: Does the reward shaping function increase or decrease the variance of policy updates and final performance across multiple random seeds?
4. **Cumulative Unshaped Reward**: Does training on shaped rewards yield an agent that performs better, worse, or equal on the *original unshaped target task objective* when evaluated deterministically?
5. **Robustness to State Variations**: Does the policy trained under a specific reward shaper generalize robustly to varied environment initialization distributions compared to the baseline?
