# ADR-002: Gymnasium Selection

## Status
Approved

## Context
We need a standard interface to define reinforcement learning environments. Historically, OpenAI Gym was the industry standard. However, Gym was officially deprecated in 2022. Using the legacy Gym library results in compatibility issues (particularly with newer versions of NumPy, which deprecated old types used by Gym) and lacks support for modern environment wrappers.

## Alternatives Considered
1. **Legacy OpenAI Gym**: The original `gym` library.
   - *Trade-off*: Unmaintained since 2022, incompatible with NumPy `>= 2.0.0`, and lacks type support.
2. **Farama Gymnasium**: The official, community-maintained fork of Gym.
   - *Trade-off*: Fully backward compatible, resolves NumPy compatibility, integrates type hints, and updates `step()` to return `(terminated, truncated)` instead of a single `done` boolean.

## Decision
We select **Farama Gymnasium** as our environment interface.

## Consequences
- **Benefits**:
  - Full compatibility with current deep learning stack libraries.
  - Standardized transition step return `(obs, reward, terminated, truncated, info)`, allowing cleaner separation between terminal physics failure (`terminated`) and arbitrary episode timeout (`truncated`).
  - Active maintenance and community support.
- **Trade-offs**:
  - Requires updating environment monitor calls and vectorized environment loaders to conform to Gymnasium's API (already handled in SB3 which fully supports Gymnasium).
