# Reward Shaping Strategies for Proximal Policy Optimization (PPO): A Comparative Experimental Study

An open-source, modular, and research-grade reinforcement learning framework designed to systematically compare and benchmark different **reward shaping strategies** for Proximal Policy Optimization (PPO). 

This project keeps environment physics, neural network initialization, and hyperparameters constant across all conditions to ensure unbiased statistical evaluation. It is built to support scientific publication, generating learning curves with shaded 95% confidence intervals, training loss analysis, and comparative performance summaries.

---

## Research Concept: What is Reward Shaping?

In reinforcement learning, the environment's reward function defines the target task objective. However, many real-world environments present **sparse rewards** (e.g., the agent only receives a reward when it achieves the final goal). This makes learning extremely sample-inefficient, as the agent must explore a vast space of actions before receiving any training signal.

**Reward Shaping** is a technique where auxiliary rewards are added to the environment's raw reward signal to guide the agent's exploration:
$$R_{shaped}(s, a, s') = R_{original}(s, a, s') + F(s, a, s')$$
where $F(s, a, s')$ is the shaping term. 

However, arbitrary shaping can alter the optimal policy (a phenomenon known as policy subversion). To prevent this, research often utilizes **Potential-Based Reward Shaping (PBRS)**, where:
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
where $\Phi(s)$ is a potential function mapping states to real numbers, and $\gamma$ is the discount factor. PBRS is mathematically proven to preserve the set of optimal policies while speeding up convergence.

---

## Project Structure

The codebase is structured to decouple environment dynamics, training logic, shaping functions, and statistical visualization.

```
reward-shaping-ppo/
├── configs/
│   ├── cartpole_baseline.yaml       # Full 5-seed, 100k step benchmark settings
│   └── cartpole_short.yaml          # 1-seed, 10k step verification run settings
├── environments/
│   ├── __init__.py                  # Wrapper exports
│   └── wrapper.py                   # RewardShapingWrapper intercepts env transitions
├── reward_functions/
│   ├── __init__.py                  # Factory method for instantiating shapers
│   ├── base.py                      # Abstract Base Class for RewardShaper
│   └── identity.py                  # IdentityRewardShaper (unmodified baseline)
├── callbacks/
│   ├── __init__.py                  # Callback exports
│   └── logging_callback.py          # Custom callback for TensorBoard logging
├── experiments/
│   ├── __init__.py                  # Runner exports
│   └── runner.py                    # ExperimentRunner orchestrates multiple seeds
├── analysis/
│   ├── __init__.py                  # Stats exports
│   └── statistics.py                # Computes means, medians, CIs across seeds
├── utils/
│   ├── __init__.py
│   ├── config.py                    # Strong-typed YAML config parser
│   ├── plotting.py                  # Publication-grade plotting in PNG & PDF
│   └── reproducibility.py           # Random seed manager
├── logs/                            # Output directory for TensorBoard tfevents
├── models/                          # Output directory for model checkpoints (.zip)
├── results/                         # Output directory for CSV logs & JSON summaries
├── plots/                           # Output directory for figures
├── requirements.txt                 # Pin dependencies
├── README.md                        # Documentation
└── main.py                          # primary framework CLI entrypoint
```

---

## Installation

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd reward-shaping-ppo
   ```

2. **Install the dependencies** listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Stable-Baselines3 supports Python 3.8 to 3.11. The framework is tested on Python 3.11.8.*

---

## How to Run Experiments

The framework uses `main.py` as its central command-line interface. You can configure execution via command-line arguments:

```bash
python main.py --config <path_to_yaml> --mode <execution_mode>
```

### Modes of Operation:
- `train`: Runs the PPO training loop across all configured seeds.
- `analyze`: Reads the saved CSV logs and computes descriptive statistics.
- `plot`: Generates training curves, evaluation curves, and loss figures.
- `all` *(default)*: Sequentially runs training, computes statistics, and generates plots.

### Verification Run (Quick Check):
Run a short, 1-seed trial for 10,000 steps to verify that the environment, logging, and plotting function correctly on your machine:
```bash
python main.py --config configs/cartpole_short.yaml --mode all
```

### Full Benchmark Run:
Run the complete baseline suite (5 seeds, 100,000 steps each):
```bash
python main.py --config configs/cartpole_baseline.yaml --mode all
```

---

## How to Add New Reward Shaping Strategies

Adding a new reward shaping strategy requires no changes to the PPO agent or environment wrappers. Follow these steps:

### Step 1: Implement the Shaper
Create a new file in `reward_functions/` (e.g., `reward_functions/distance.py`) and inherit from the `RewardShaper` abstract base class:

```python
from typing import Any, Dict
from reward_functions.base import RewardShaper

class DistanceRewardShaper(RewardShaper):
    def __init__(self, target_distance: float = 0.5):
        self.target_distance = target_distance
        self.initial_pos = None

    def reset(self, initial_state: Any, info: Dict[str, Any]) -> None:
        # Reset episode-specific variables
        self.initial_pos = initial_state[0]

    def shape_reward(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: Dict[str, Any]
    ) -> float:
        # State: [cart_position, cart_velocity, pole_angle, pole_velocity]
        cart_pos = next_state[0]
        # Calculate custom shaping term
        shaping_term = -abs(cart_pos - self.target_distance)
        return reward + shaping_term
```

### Step 2: Register the Shaper in the Factory
Import and register your class inside `reward_functions/__init__.py`:

```diff
from reward_functions.base import RewardShaper
from reward_functions.identity import IdentityRewardShaper
+from reward_functions.distance import DistanceRewardShaper

def get_reward_shaper(strategy_name: str, params: Optional[Dict[str, Any]] = None) -> RewardShaper:
    if params is None:
        params = {}
    name = strategy_name.lower().strip()
    
    if name == "identity":
        return IdentityRewardShaper()
+   elif name == "distance":
+       return DistanceRewardShaper(**params)
    else:
        raise ValueError(f"Unknown reward shaping strategy: '{strategy_name}'")
```

### Step 3: Configure and Train
Create a new configuration file (e.g., `configs/cartpole_distance.yaml`) and update the `reward_shaping` block:
```yaml
reward_shaping:
  strategy: "distance"
  params:
    target_distance: 0.8
```
Run training:
```bash
python main.py --config configs/cartpole_distance.yaml --mode all
```

---

## Reproducibility Standards

To guarantee identical results when re-running benchmarks:
- **Environment Seed**: Action and observation spaces are seeded directly at the start of each episode.
- **Model Seed**: The PyTorch neural network policy parameters are initialized using a fixed seed, and the action selection distribution is seeded inside Stable-Baselines3.
- **PyTorch Backends**: Set `deterministic: true` in the configuration files to force PyTorch's CuDNN backend to use deterministic convolution algorithms. Note: This can lead to a slight performance slowdown on complex network topologies, but ensures bit-level reproducibility.
