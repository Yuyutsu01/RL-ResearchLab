import os
import tempfile

import yaml

from utils.config import Config


def test_config_loader():
    """Verifies that the YAML config manager correctly parses files into typed dataclasses."""
    test_yaml = {
        "experiment": {
            "name": "test_exp",
            "env_id": "CartPole-v1",
            "total_timesteps": 1000,
            "eval_freq": 200,
            "eval_episodes": 2,
            "checkpoint_freq": 500,
            "seeds": [1, 2],
            "reproducibility": {"deterministic": True, "benchmark": False},
        },
        "ppo": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 32,
            "policy_kwargs": {"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
        },
        "reward_shaping": {"strategy": "identity", "params": {"factor": 1.0}},
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_yaml, f)
        temp_path = f.name

    try:
        config = Config.from_yaml(temp_path)

        # Verify Experiment Configurations
        assert config.experiment.name == "test_exp"
        assert config.experiment.env_id == "CartPole-v1"
        assert config.experiment.total_timesteps == 1000
        assert config.experiment.seeds == [1, 2]
        assert config.experiment.reproducibility.deterministic is True

        # Verify PPO Hyperparameters
        assert config.ppo.learning_rate == 0.001
        assert config.ppo.gamma == 0.99
        assert config.ppo.batch_size == 32
        assert config.ppo.policy_kwargs["net_arch"]["pi"] == [32, 32]

        # Verify Reward Shaping Configuration
        assert config.reward_shaping.strategy == "identity"
        assert config.reward_shaping.params == {"factor": 1.0}

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
