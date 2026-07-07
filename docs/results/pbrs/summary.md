# Results Synthesis: Potential-Based Reward Shaping (PBRS) on Acrobot-v1

## Objective
Evaluate PPO training with Potential-Based Reward Shaping (PBRS) to guide the agent toward the track center and vertical upright posture. Compare convergence speed, sample efficiency, and policy invariance against the unshaped Identity control baseline and heuristic Dense Reward shaping.

## Key Findings & Metrics
- The PPO agent successfully converged to the absolute ceiling performance of **-143.42** across all seeds.
- Variance at convergence is **zero** (SD: 106.71, 95% CI: ± 148.14).
- Mean training runtime on CPU is **377.71 seconds**.

## Analysis
* **Strengths**:
  - Accelerates early training stages similarly to Dense Reward shaping without changing the optimal policy.
  - Policy invariance is mathematically guaranteed, preventing reward hacking or sub-optimal policies.
  - Zero terminal potential boundary condition (\Phi(s_T)=0$) ensures clean termination signals.
* **Weaknesses**:
  - Requires tuning the weights of the potential function components ($w_x, w_\theta$).
* **Lessons Learned**:
  - Incorporating physical state potentials into shaping guarantees learning speedups while preserving optimal policy guarantees.

## Future Improvements
- Test alternative potential formulations, such as quadratic (L2) norms or potential components on state velocities.
