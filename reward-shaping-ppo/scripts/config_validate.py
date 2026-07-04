#!/usr/bin/env python3
"""Config Schema Validator for CI gates."""

import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config

VALID_TOP_LEVEL_KEYS = {"experiment", "ppo", "reward_shaping"}
VALID_EXPERIMENT_KEYS = {
    "name",
    "env_id",
    "total_timesteps",
    "eval_freq",
    "eval_episodes",
    "checkpoint_freq",
    "seeds",
    "reproducibility",
}
VALID_PPO_KEYS = {
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "policy_kwargs",
    "device",
}
VALID_REWARD_SHAPING_KEYS = {"strategy", "params"}
VALID_REPRODUCIBILITY_KEYS = {"deterministic", "benchmark"}


def validate_config(filepath: str) -> list[str]:
    errors = []
    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    if not isinstance(data, dict):
        return [f"Expected a YAML mapping, got {type(data).__name__}"]

    unknown_top = set(data.keys()) - VALID_TOP_LEVEL_KEYS
    if unknown_top:
        errors.append(f"Unknown top-level keys: {unknown_top}")

    if "experiment" in data and isinstance(data["experiment"], dict):
        unknown_exp = set(data["experiment"].keys()) - VALID_EXPERIMENT_KEYS
        if unknown_exp:
            errors.append(f"Unknown 'experiment' keys: {unknown_exp}")

        reprod = data["experiment"].get("reproducibility", {})
        if isinstance(reprod, dict):
            unknown_reprod = set(reprod.keys()) - VALID_REPRODUCIBILITY_KEYS
            if unknown_reprod:
                errors.append(f"Unknown 'experiment.reproducibility' keys: {unknown_reprod}")

    if "ppo" in data and isinstance(data["ppo"], dict):
        unknown_ppo = set(data["ppo"].keys()) - VALID_PPO_KEYS
        if unknown_ppo:
            errors.append(f"Unknown 'ppo' keys: {unknown_ppo}")

    if "reward_shaping" in data and isinstance(data["reward_shaping"], dict):
        unknown_rs = set(data["reward_shaping"].keys()) - VALID_REWARD_SHAPING_KEYS
        if unknown_rs:
            errors.append(f"Unknown 'reward_shaping' keys: {unknown_rs}")

    try:
        Config.from_yaml(filepath)
    except Exception as e:
        errors.append(f"Config.from_yaml() raised: {e}")

    return errors


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
        files = glob.glob(os.path.join(configs_dir, "*.yaml")) + glob.glob(os.path.join(configs_dir, "*.yml"))

    if not files:
        sys.exit(0)

    total_errors = 0
    for filepath in sorted(files):
        basename = os.path.basename(filepath)
        errors = validate_config(filepath)
        if errors:
            print(f"FAIL  {basename}")
            for err in errors:
                print(f"  ✗ {err}")
            total_errors += len(errors)
        else:
            print(f"  OK  {basename}")

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
