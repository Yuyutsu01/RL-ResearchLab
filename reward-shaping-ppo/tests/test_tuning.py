import os
import json
import yaml
import pytest
from analysis.tune_analysis import determine_best_config, export_optimized_config

@pytest.fixture
def dummy_tuning_setup(tmp_path):
    # Create results/TestEnv-v0 directory
    results_dir = tmp_path / "results" / "TestEnv-v0"
    os.makedirs(results_dir, exist_ok=True)

    # Create configs folder
    configs_dir = tmp_path / "configs"
    os.makedirs(configs_dir, exist_ok=True)

    # Write a dummy baseline yaml
    dummy_yaml_content = {
        "experiment": {
            "name": "test_env_baseline",
            "env_id": "TestEnv-v0",
            "total_timesteps": 1000,
            "seeds": [42]
        },
        "ppo": {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
        },
        "reward_shaping": {
            "strategy": "identity",
            "params": {}
        }
    }

    src_config_file = configs_dir / "acrobot_identity.yaml"
    with open(src_config_file, "w") as f:
        yaml.dump(dummy_yaml_content, f)

    # Mock mapping internally
    import analysis.tune_analysis
    analysis.tune_analysis.env_config_map = {
        "TestEnv-v0": "acrobot_identity.yaml"
    }

    # Write dummy summary JSON
    summary_data = {
        "learning_rate": {
            "3e-04": {"mean_reward": -200.0, "mean_auc": 0.1},
            "1e-04": {"mean_reward": -100.0, "mean_auc": 0.2},  # Best
            "5e-05": {"mean_reward": -150.0, "mean_auc": 0.15}
        },
        "batch_size": {
            "64": {"mean_reward": -100.0, "mean_auc": 0.2},
            "128": {"mean_reward": -80.0, "mean_auc": 0.3},    # Best
            "256": {"mean_reward": -90.0, "mean_auc": 0.25}
        },
        "gamma": {
            "0.99": {"mean_reward": -100.0, "mean_auc": 0.2},
            "0.995": {"mean_reward": -100.0, "mean_auc": 0.2},
            "0.999": {"mean_reward": -100.0, "mean_auc": 0.2}
        },
        "gae_lambda": {
            "0.95": {"mean_reward": -100.0, "mean_auc": 0.2},
            "0.97": {"mean_reward": -100.0, "mean_auc": 0.2},
            "0.99": {"mean_reward": -100.0, "mean_auc": 0.2}
        },
        "ent_coef": {
            "0.0": {"mean_reward": -100.0, "mean_auc": 0.2},
            "0.01": {"mean_reward": -100.0, "mean_auc": 0.2},
            "0.02": {"mean_reward": -100.0, "mean_auc": 0.2}
        },
        "clip_range": {
            "0.2": {"mean_reward": -100.0, "mean_auc": 0.2},
            "0.15": {"mean_reward": -100.0, "mean_auc": 0.2},
            "0.1": {"mean_reward": -100.0, "mean_auc": 0.2}
        },
        "net_arch": {
            "64x64": {"mean_reward": -100.0, "mean_auc": 0.2},
            "128x128": {"mean_reward": -50.0, "mean_auc": 0.4}, # Best
            "256x256": {"mean_reward": -60.0, "mean_auc": 0.35}
        }
    }

    summary_path = results_dir / "tuning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f)

    return tmp_path

class TestTuningAnalysis:
    def test_determine_best_config(self, dummy_tuning_setup):
        optimized_ppo = determine_best_config("TestEnv-v0", base_dir=str(dummy_tuning_setup))

        # Verify optimized parameters selection
        assert optimized_ppo["learning_rate"] == 1e-4
        assert optimized_ppo["batch_size"] == 128
        assert optimized_ppo["policy_kwargs"]["net_arch"] == {"pi": [128, 128], "vf": [128, 128]}
        assert optimized_ppo["gamma"] == 0.99
        assert optimized_ppo["gae_lambda"] == 0.95
        assert optimized_ppo["ent_coef"] == 0.0
        assert optimized_ppo["clip_range"] == 0.2

    def test_export_optimized_config(self, dummy_tuning_setup):
        dest_config_path = export_optimized_config("TestEnv-v0", base_dir=str(dummy_tuning_setup))

        assert os.path.exists(dest_config_path)
        with open(dest_config_path, "r") as f:
            optimized_config = yaml.safe_load(f)

        assert optimized_config["experiment"]["name"] == "testenv_v0_optimized_study"
        assert optimized_config["ppo"]["learning_rate"] == 1e-4
        assert optimized_config["ppo"]["batch_size"] == 128
        assert optimized_config["ppo"]["policy_kwargs"]["net_arch"] == {"pi": [128, 128], "vf": [128, 128]}
