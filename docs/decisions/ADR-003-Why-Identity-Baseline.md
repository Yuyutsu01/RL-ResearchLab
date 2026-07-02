# ADR-003: Core Identity Baseline Choice

## Status
Approved

## Context
When investigating the effects of reward shaping, we need a rigorous control baseline to measure whether a given shaping strategy actually speeds up convergence, improves sample efficiency, or degrades policy performance. Without a control condition where the reward function is unmodified, we cannot quantify the relative speedup factor or evaluate whether the shaping function is causing policy subversion.

## Alternatives Considered
1. **No Control (Only compare custom shapers)**: Comparing Dense vs. PBRS directly without an unshaped run.
   - *Trade-off*: We cannot determine whether *either* shaper is performing better or worse than default training, making speedup claims scientifically unsubstantiated.
2. **Identity Baseline (Unaltered Pass-Through)**: Running standard PPO directly on the raw environment reward.
   - *Trade-off*: Requires extra computation resources (running 5 seeds on the control baseline), but yields a statistically sound, unmodified baseline.

## Decision
We establish the **Identity Reward** (unmodified pass-through) as our baseline condition. All subsequent reward shaping strategies will be benchmarked against this baseline run using identical hyperparameters.

## Consequences
- **Benefits**:
  - Provides a scientifically rigorous control group to calculate "Speed-up Factors" and evaluate learning rates.
  - Allows us to easily verify whether a shaping function causes policy subversion by comparing performance on the target objective.
- **Trade-offs**:
  - Requires additional training time (5 seeds, 100k steps) to run the baseline, which is minor given the 15-minute CPU run duration.
