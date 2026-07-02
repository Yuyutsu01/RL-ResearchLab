# ADR-005: Unbiased Separate Evaluation Environments

## Status
Approved

## Context
If an agent is trained on a shaped reward (e.g. including auxiliary signals guiding velocity or center balance), its training curve logs the shaped reward. If we evaluate policy performance on the same shaped environment, we are measuring progress on a proxy objective. This introduces evaluation bias and masks cases of policy subversion (where the agent scores highly on the proxy reward but fails the original task).

## Alternatives Considered
1. **Shared Training & Evaluation Environments**: Evaluating on the same shaped environment.
   - *Trade-off*: We cannot measure how well the agent performs on the *original* task. If the shaping reward has a different range (e.g. large penalties), comparisons between different strategies become mathematically mismatched.
2. **Decoupled Evaluation Environments**: Evaluating policies strictly on an environment wrapped with the `IdentityRewardShaper` (raw reward).
   - *Trade-off*: Requires keeping two separate environment instances (Train and Eval) and running evaluations deterministically, which is standard in RL benchmarking.

## Decision
We choose to use **decoupled training and evaluation environments**. The evaluation environment must always use the raw, unshaped reward structure.

## Consequences
- **Benefits**:
  - Unbiased measurement of policy performance on the target physics task.
  - Allows direct, mathematically aligned comparisons of learning curves across completely different shaping functions.
  - Immediately exposes policy subversion (the agent gets high training shaped rewards but flat evaluation rewards).
- **Trade-offs**:
  - Dual environment instantiation memory footprint (negligible for classic control).
