import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ReproducibilityConfig:
    """Configuration for seeds and PyTorch backends."""

    deterministic: bool = True
    benchmark: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment metadata, checkpoints, and paths."""

    name: str
    env_id: str
    total_timesteps: int
    eval_freq: int = 5000
    eval_episodes: int = 10
    checkpoint_freq: int = 20000
    seeds: List[int] = field(default_factory=lambda: [42])
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )


@dataclass
class PPOHyperparameters:
    """Hyperparameters passed directly to Stable-Baselines3 PPO."""

    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"net_arch": dict(pi=[64, 64], vf=[64, 64])}
    )
    device: str = "cpu"


@dataclass
class RewardShapingConfig:
    """Configuration for the active reward shaping strategy."""

    strategy: str = "identity"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """The root configuration object containing all sub-configs."""

    experiment: ExperimentConfig
    ppo: PPOHyperparameters
    reward_shaping: RewardShapingConfig

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        """
        Loads configuration from a YAML file and parses it into typed dataclasses.

        Args:
            file_path: The absolute or relative path to the YAML file.

        Returns:
            An instance of Config containing populated configurations.
        """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Parse experiment config
        exp_data = data.get("experiment", {})
        reprod_data = exp_data.get("reproducibility", {})
        reprod_config = ReproducibilityConfig(
            deterministic=reprod_data.get("deterministic", True),
            benchmark=reprod_data.get("benchmark", False),
        )
        experiment_config = ExperimentConfig(
            name=exp_data.get("name", "experiment"),
            env_id=exp_data.get("env_id", "CartPole-v1"),
            total_timesteps=exp_data.get("total_timesteps", 100000),
            eval_freq=exp_data.get("eval_freq", 5000),
            eval_episodes=exp_data.get("eval_episodes", 10),
            checkpoint_freq=exp_data.get("checkpoint_freq", 20000),
            seeds=exp_data.get("seeds", [42]),
            reproducibility=reprod_config,
        )

        # Parse PPO hyperparameters
        ppo_data = data.get("ppo", {})
        ppo_config = PPOHyperparameters(
            learning_rate=float(ppo_data.get("learning_rate", 0.0003)),
            n_steps=int(ppo_data.get("n_steps", 2048)),
            batch_size=int(ppo_data.get("batch_size", 64)),
            n_epochs=int(ppo_data.get("n_epochs", 10)),
            gamma=float(ppo_data.get("gamma", 0.99)),
            gae_lambda=float(ppo_data.get("gae_lambda", 0.95)),
            clip_range=float(ppo_data.get("clip_range", 0.2)),
            ent_coef=float(ppo_data.get("ent_coef", 0.0)),
            vf_coef=float(ppo_data.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo_data.get("max_grad_norm", 0.5)),
            policy_kwargs=ppo_data.get(
                "policy_kwargs", {"net_arch": dict(pi=[64, 64], vf=[64, 64])}
            ),
            device=ppo_data.get("device", "cpu"),
        )

        # Parse reward shaping config
        shaping_data = data.get("reward_shaping", {})
        shaping_config = RewardShapingConfig(
            strategy=shaping_data.get("strategy", "identity"),
            params=shaping_data.get("params", {}),
        )

        return cls(
            experiment=experiment_config, ppo=ppo_config, reward_shaping=shaping_config
        )
