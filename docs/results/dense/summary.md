# Results Synthesis: Dense Reward on CartPole-v1

## Objective
Evaluate PPO training with a dense reward function targeting cart centering and pole balancing. Compare convergence speed, sample efficiency, and stability against the unshaped Identity control baseline.

## Key Findings & Metrics
- The PPO agent successfully converged to the absolute ceiling performance of **500.00** across seeds.
- Variance at convergence is **zero** (SD: 0.00, 95% CI: ± 0.00).
- Mean training runtime on CPU is **191.23 seconds**.
- The speedup to reach intermediate thresholds is significant.

## Analysis
* **Strengths**:
  - Dramatically improves sample efficiency, reducing convergence timesteps to reach stable balancing.
  - Smooth reward gradient mitigates early-stage random walk exploration.
* **Weaknesses**:
  - Highly wobbly cart behavior at convergence compared to PBRS, due to linear penalty weights.
* **Lessons Learned**:
  - Dense shaping is highly effective at guiding exploration but requires careful weight balancing to prevent subversion of target constraints.

## Future Improvements
- Move to Potential-Based Reward Shaping (PBRS) to guarantee policy preservation while retaining convergence speedups.
