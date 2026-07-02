# System Design Overview

The PPO Reward Shaping Research Framework is built to support modularity, reproducibility, and automation. By isolating environment dynamics, agent training, reward modifications, and logging, the framework allows researchers to introduce new environments or reward shapers without modifying the core experiment orchestration codebase.

---

## Architecture Design Principles

1. **SOLID Principles**:
   - *Single Responsibility*: Classes like `RewardShapingWrapper` focus strictly on state transition metrics, `RewardShaper` subclasses focus on shaping math, and `ExperimentRunner` handles orchestrations.
   - *Open/Closed*: The framework is open to adding new reward functions (via the shaper factory) but closed to modification of the training loop.
2. **Deterministic Execution**:
   - Random seeds are systematically propagated through Python, NumPy, PyTorch, Gymnasium, and Stable-Baselines3.
3. **Decoupled Evaluator**:
   - The evaluation environment is wrapped using an `IdentityRewardShaper`, preventing training proxy values from skewing true task metrics.
4. **Structured Research Outputs**:
   - Every execution compiles structured logs, metadata JSONs, model checkpoints, and high-quality charts inside the `results/` and `plots/` directories, which are then archived inside the `docs/` system by the `AutoDocManager` engine.

---

## Core Component Modules

```
                        +----------------------------+
                        |      main.py CLI           |
                        +----------------------------+
                                      |
                                      v
                        +----------------------------+
                        |      ExperimentRunner      |
                        +----------------------------+
                               /              \
                              /                \
                             v                  v
                 +-------------------+  +-------------------+
                 |   Training Env    |  |  Evaluation Env   |
                 | (Shaped Wrapper)  |  | (Identity Wrapper)|
                 +-------------------+  +-------------------+
                          |                       |
                          | (actions/states)      | (eval actions/states)
                          v                       v
                 +------------------------------------------+
                 |          Stable-Baselines3 PPO           |
                 +------------------------------------------+
                        /                           \
                       v                             v
            +----------------------+     +----------------------+
            |    Monitor Log CSV   |     | ResearchLoggingCall  |
            +----------------------+     +----------------------+
                       \                             /
                        v                           v
            +-----------------------------------------------+
            |          Experiment Statistics & Plots        |
            +-----------------------------------------------+
                                      |
                                      v
            +-----------------------------------------------+
            |         AutoDocManager Archive Engine         |
            +-----------------------------------------------+
                                      |
                                      v
            +-----------------------------------------------+
            |       docs/ Experiments & Evidence Catalog     |
            +-----------------------------------------------+
```
