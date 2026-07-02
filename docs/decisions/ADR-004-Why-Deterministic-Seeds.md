# ADR-004: Standardized Multi-Random Seeds

## Status
Approved

## Context
Reinforcement learning is notoriously sensitive to initialization seeds. A single run might succeed or fail purely due to network weight initialization or starting state distributions. To draw scientifically valid conclusions, we must evaluate strategies across a range of random seeds. Additionally, to ensure other researchers can duplicate our results, execution must be completely deterministic when re-run.

## Alternatives Considered
1. **Dynamic Random Execution (No Seeding)**: Allowing runs to use clock-time randomness.
   - *Trade-off*: Results will vary on every execution, making reproducibility impossible and making comparison highly noisy.
2. **Single Seed Execution**: Seeding only one seed (e.g. 42).
   - *Trade-off*: Lacks statistical power. A strategy might look superior purely due to a lucky initialization on seed 42.
3. **Multi-Seed Standardized Determinism**: Running experiments across a fixed array of seeds (`[42, 43, 44, 45, 46]`) with PyTorch deterministic backends enabled.
   - *Trade-off*: Ensures robust statistics and exact bit-level reproducibility, though forcing PyTorch's backends to be deterministic can slightly slow down computation.

## Decision
We enforce a standardized **5-seed deterministic execution** protocol. All experiments are evaluated on seeds `[42, 43, 44, 45, 46]` with deterministic library seeds and PyTorch backends enabled.

## Consequences
- **Benefits**:
  - Eliminates "cherry-picking" of lucky seed runs.
  - Allows computation of 95% confidence intervals and standard errors of the mean (SEM), satisfying publication standards.
  - Guarantees that any researcher re-running the command gets identical data, curves, and statistics.
- **Trade-offs**:
  - Slight performance slowdown on GPUs (minimized in our case since we run on CPU).
  - Multiplies execution time by the number of seeds (5x), which is acceptable given our CPU optimizations.
