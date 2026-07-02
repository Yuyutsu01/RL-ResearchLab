# ADR-006: Modular Decoupled Wrapper Interface

## Status
Approved

## Context
We need a reward shaping architecture that is easy to extend. If the reward shaping logic is hardcoded into the agent implementation (like modifying PPO's loss or policy step) or into the Gymnasium environment files, adding a new reward strategy requires editing core code. This increases risk of regression bugs and complicates maintaining a unified benchmarking suite.

## Alternatives Considered
1. **Hardcoding into PPO**: Intercepting and modifying step values inside Stable-Baselines3's library files.
   - *Trade-off*: Intrusive, hard to maintain, and breaks when upgrading libraries.
2. **Editing Environment files**: Designing a custom `CartPole` class with built-in reward shaping branches.
   - *Trade-off*: Inflexible. If we want to evaluate on a new environment (like `LunarLander-v2`), we must duplicate all the reward shaping code inside the new environment class.
3. **Decoupled Wrapper & Shaper API**: Implementing a custom Gymnasium `Wrapper` that accepts an abstract `RewardShaper` component.
   - *Trade-off*: Clean, modular, and reusable. We can wrap *any* Gymnasium environment with *any* `RewardShaper` dynamically.

## Decision
We select the **Decoupled Wrapper & Shaper API** pattern. The `RewardShapingWrapper` class wraps a Gymnasium environment and delegates reward calculation to an instance of `RewardShaper`.

## Consequences
- **Benefits**:
  - Open/Closed principle: Adding a new strategy requires creating a new `RewardShaper` class and registering it in the factory, without editing training or environment code.
  - Generality: The wrapper can wrap any Gymnasium environment (e.g. `LunarLander-v2`), making cross-environment benchmarking straightforward.
- **Trade-offs**:
  - The shaper must receive transitions using the generic state/action signatures, which is supported by using type annotations and shape checking.
