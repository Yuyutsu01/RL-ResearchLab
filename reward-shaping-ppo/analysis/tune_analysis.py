import os
import json
import yaml
from typing import Any, Dict

env_config_map = {
    "CartPole-v1": "cartpole_baseline.yaml",
    "MountainCar-v0": "mountaincar_identity.yaml",
    "Acrobot-v1": "acrobot_identity.yaml",
    "LunarLander-v3": "lunarlander_identity.yaml"
}

def determine_best_config(env_id: str, base_dir: str = ".") -> Dict[str, Any]:
    """
    Parses tuning_summary.json, identifies the best-performing value for
    each coordinate sweep axis, and returns the optimized hyperparameters dictionary.
    """
    summary_path = os.path.join(base_dir, "results", env_id, "tuning_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Tuning summary not found at: {summary_path}")

    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    # Reference default config values as fallback
    best_params = {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "clip_range": 0.2,
        "net_arch": "64x64"
    }

    # Sweep axes in the summary file
    axes = ["learning_rate", "batch_size", "gamma", "gae_lambda", "ent_coef", "clip_range", "net_arch"]

    for axis in axes:
        if axis not in summary_data:
            continue

        options = summary_data[axis]
        best_val = None
        best_reward = float("-inf")
        best_auc = float("-inf")

        for val_str, stats in options.items():
            mean_reward = stats.get("mean_reward", float("-inf"))
            mean_auc = stats.get("mean_auc", float("-inf"))

            # Convert string key back to float/int where appropriate
            try:
                if "." in val_str or "e" in val_str:
                    val = float(val_str)
                else:
                    val = int(val_str)
            except ValueError:
                val = val_str # Keeps string for net_arch

            # Check if this option is better (maximize reward, tie-break on AUC)
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_auc = mean_auc
                best_val = val
            elif mean_reward == best_reward and mean_auc > best_auc:
                best_auc = mean_auc
                best_val = val

        if best_val is not None:
            best_params[axis] = best_val

    # Convert best parameter configuration to dictionary
    net_arch_map = {
        "64x64": dict(pi=[64, 64], vf=[64, 64]),
        "128x128": dict(pi=[128, 128], vf=[128, 128]),
        "256x256": dict(pi=[256, 256], vf=[256, 256])
    }

    optimized_ppo = {
        "learning_rate": float(best_params["learning_rate"]),
        "batch_size": int(best_params["batch_size"]),
        "gamma": float(best_params["gamma"]),
        "gae_lambda": float(best_params["gae_lambda"]),
        "ent_coef": float(best_params["ent_coef"]),
        "clip_range": float(best_params["clip_range"]),
        "policy_kwargs": {"net_arch": net_arch_map[best_params["net_arch"]]}
    }

    return optimized_ppo

def export_optimized_config(env_id: str, base_dir: str = ".") -> str:
    """
    Determines the best configuration for the env and exports it toconfigs/<env_id_clean>_optimized.yaml.
    """
    optimized_ppo = determine_best_config(env_id, base_dir)

    config_filename = env_config_map.get(env_id)
    if not config_filename:
        raise ValueError(f"Unknown environment baseline configuration file for env: {env_id}")

    src_config_path = os.path.join(base_dir, "configs", config_filename)
    if not os.path.exists(src_config_path):
        raise FileNotFoundError(f"Source baseline config not found at: {src_config_path}")

    with open(src_config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Update configurations
    config_dict["experiment"]["name"] = f"{env_id.lower().replace('-', '_')}_optimized_study"
    config_dict["experiment"]["seeds"] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51] # Restore full seeds study list
    config_dict["ppo"]["device"] = "cuda"

    # Merge optimized parameters
    for k, v in optimized_ppo.items():
        config_dict["ppo"][k] = v

    # Write optimized config YAML
    env_clean = env_id.lower().replace("-", "_")
    dest_filename = f"{env_clean}_optimized.yaml"
    dest_config_path = os.path.join(base_dir, "configs", dest_filename)

    with open(dest_config_path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False)

    print(f"Exported optimized baseline YAML configuration to: {dest_config_path}")
    return dest_config_path
