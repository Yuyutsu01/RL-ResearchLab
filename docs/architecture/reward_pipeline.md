# Reward Interception Pipeline

The framework uses a wrapper-interception pattern to modify rewards. This document details the step-by-step logic of the reward interception pipeline.

---

## The Transition Cycle

During training, the PPO agent executes a step in the environment. The `RewardShapingWrapper` intercepts this interaction as follows:

```
                  PPO Agent selects Action (a_t)
                                |
                                v
                  RewardShapingWrapper.step(a_t)
                                |
                                v
                  Underlying Gymnasium Environment
                  returns (s_{t+1}, r_{original}, term, trunc, info)
                                |
                                v
                  Calculate State Transition Variables:
                  - state: s_t (cached from previous step or reset)
                  - action: a_t
                  - reward: r_{original}
                  - next_state: s_{t+1}
                  - done: term or trunc
                  - info: environment diagnostic dict
                                |
                                v
                  Invoke: RewardShaper.shape_reward(...)
                  returns r_{shaped}
                                |
                                v
                  Update Running Episode Metrics:
                  - episode_original_reward += r_{original}
                  - episode_shaped_reward += r_{shaped}
                  - episode_length += 1
                                |
                                v
                  Populate Step Diagnostics:
                  - info["step_original_reward"] = r_{original}
                  - info["step_shaped_reward"] = r_{shaped}
                                |
                                v
                  IF done IS True:
                  - info["original_reward"] = episode_original_reward
                  - info["shaped_reward"] = episode_shaped_reward
                  - info["episode_length"] = episode_length
                                |
                                v
                  Update Caches:
                  - s_t = s_{t+1}
                                |
                                v
                  Return (s_{t+1}, r_{shaped}, term, trunc, info)
```

---

## Step vs. Episode Logging

1. **Step-Level Interception**:
   - Every step exposes `info["step_original_reward"]` and `info["step_shaped_reward"]`.
   - The wrapped reward `r_{shaped}` is returned to PPO, which uses it for Generalized Advantage Estimation (GAE) and policy updates.
2. **Episode-Level Interception**:
   - Only on terminal steps (`terminated or truncated`), the wrapper injects the final cumulative original and shaped episode returns into the `info` dictionary.
   - The `Monitor` wrapper (wrapping the environment above the shaping wrapper) detects `done = True`, reads these values using `info_keywords=("original_reward", "shaped_reward")`, and automatically writes them as a single row in the `monitor.csv` output log.
