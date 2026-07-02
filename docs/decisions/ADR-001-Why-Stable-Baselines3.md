# ADR-001: Stable-Baselines3 Selection

## Status
Approved

## Context
We need a robust, reliable, and thoroughly tested implementation of Proximal Policy Optimization (PPO) as our agent model. Writing PPO from scratch is a valuable learning exercise but carries a high risk of subtle optimization bugs (e.g. incorrect advantage calculations, improper gradient clipping, or policy entropy weight misalignments) that can obscure the comparison of reward shaping strategies.

## Alternatives Considered
1. **Custom PPO Implementation**: Building PPO from scratch in PyTorch.
   - *Trade-off*: High risk of implementation bugs, lacks standardized optimizations (like orthogonal initialization), and consumes significant development time.
2. **Ray / RLlib**: A powerful distributed reinforcement learning library.
   - *Trade-off*: Extremely high complexity, heavy package dependencies, and overkill for single-node CPU/GPU classical control environments.
3. **Stable-Baselines3 (SB3)**: A popular, PyTorch-based, modular, and actively maintained RL baseline library.
   - *Trade-off*: Easy to customize, provides standard logging/evaluation callbacks out of the box, and has a proven track record of correctness.

## Decision
We choose **Stable-Baselines3** as the agent optimization library. It is well-documented, widely cited in reinforcement learning research, and implements PPO with verified hyperparameter benchmarks.

## Consequences
- **Benefits**:
  - We can focus development effort entirely on the reward shaping interface and experiment pipeline.
  - Highly optimized and correct PPO implementation (incorporating standard features like advantage normalization and orthogonal policy initialization).
  - Easy integration with custom callback structures.
- **Trade-offs**:
  - We are constrained by SB3's structure and configuration parameters (which is not an issue since it supports custom policies and environments).
  - Heavy dependence on SB3 updates (pinned to version `>=2.1.0` to ensure API stability).
