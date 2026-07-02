# Component and API Reference

This document provides a reference for all framework modules, class signatures, and primary utility functions.

---

## 1. Environment Wrapper (`environments/wrapper.py`)

### `RewardShapingWrapper`
Inherits from `gymnasium.Wrapper`. Intercepts transition steps to modify reward values.
* **Constructor**:
  ```python
  def __init__(self, env: gym.Env, reward_shaper: RewardShaper): ...
  ```
* **Key Attributes**:
  - `reward_shaper`: The active `RewardShaper` instance.
  - `_current_obs`: Cache of the state before stepping.
  - `_episode_original_reward`: Running sum of raw rewards.
  - `_episode_shaped_reward`: Running sum of shaped rewards.
  - `_episode_length`: Total steps in the current episode.
* **Key Methods**:
  - `reset(self, **kwargs)`: Resets trackers, environment, and shaper. Returns `(initial_obs, info)`.
  - `step(self, action)`: Steps physics, shapes reward, accumulates stats, and returns transition tuple. Injects episode totals in `info` if `done = True`.

---

## 2. Reward Shapers (`reward_functions/`)

### `RewardShaper` (`reward_functions/base.py`)
Abstract base class defining the reward shaping interface.
* **Methods**:
  - `reset(self, initial_state: Any, info: Dict[str, Any]) -> None`: Clear/initialize episode states.
  - `shape_reward(self, state, action, reward, next_state, done, info) -> float`: Process transition and return shaped reward.

### `IdentityRewardShaper` (`reward_functions/identity.py`)
Control baseline pass-through implementation of `RewardShaper`.

### `get_reward_shaper` (`reward_functions/__init__.py`)
Factory function resolving strategies by name.
```python
def get_reward_shaper(strategy_name: str, params: Optional[Dict[str, Any]] = None) -> RewardShaper: ...
```

---

## 3. Callbacks (`callbacks/logging_callback.py`)

### `ResearchLoggingCallback`
Inherits from `stable_baselines3.common.callbacks.BaseCallback`. Extracts monitor statistics and updates TensorBoard.
* **Constructor**:
  ```python
  def __init__(self, rolling_window: int = 100, verbose: int = 0): ...
  ```
* **Key Methods**:
  - `_on_step(self) -> bool`: Executed at every environment step. Checks for completed episodes and updates rolling `deques` for logging.

---

## 4. Experiment Runner (`experiments/runner.py`)

### `ExperimentRunner`
Orchestrates training across seeds and configs.
* **Constructor**:
  ```python
  def __init__(self, config_path: str, base_dir: str = "."): ...
  ```
* **Key Methods**:
  - `run_single_seed(self, seed: int) -> Dict[str, Any]`: Seeds framework, creates folders, constructs vectorized train/eval envs, instantiates PPO model, and runs training. Saves metadata and final checkpoint.
  - `run_all(self) -> Dict[int, Dict[str, Any]]`: Iterates over configured seeds sequentially.

---

## 5. Statistical Analyzer (`analysis/statistics.py`)

### `ExperimentAnalyzer`
Aggregates and aligns monitor logs across multiple seeds.
* **Constructor**:
  ```python
  def __init__(self, env_id: str, base_dir: str = "."): ...
  ```
* **Key Methods**:
  - `load_strategy_data(self, strategy, grid_points, rolling_window)`: Loads CSV logs, aligns variable-length episodes on a step grid, and calculates mean, std, and 95% CIs.
  - `compute_summary_statistics(self, strategy, last_pct)`: Averages unshaped rewards and training times over the final fraction of training steps. Saves `summary.json`.
  - `generate_comparison_report(self, strategies)`: Compiles summary tables across strategies and saves `comparison_report.csv`.
