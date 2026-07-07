import os
import json
import yaml
import shutil
import numpy as np
from typing import Any, Dict, List
from utils.config import Config
from experiments.runner import ExperimentRunner

class HyperparameterTuner:
    """
    Manages coordinate sensitivity sweeps for PPO hyperparameters
    (Learning Rate, Batch Size, Gamma, GAE Lambda, Entropy Coefficient,
    Clip Range, and Network Architecture) on a target environment.
    """

    def __init__(self, base_config_path: str, base_dir: str = "."):
        self.base_config_path = base_config_path
        self.base_dir = base_dir
        self.base_config = Config.from_yaml(base_config_path)
        self.env_id = self.base_config.experiment.env_id

        # Use 3 random seeds for tuning budget efficiency
        self.seeds = [42, 43, 44]

        # Reference baseline PPO hyperparameters
        self.baseline_hparams = {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "net_arch": "64x64"
        }

        # Parameter axes to sweep
        self.sweep_axes = {
            "learning_rate": [3e-4, 1e-4, 5e-5],
            "batch_size": [64, 128, 256],
            "gamma": [0.99, 0.995, 0.999],
            "gae_lambda": [0.95, 0.97, 0.99],
            "ent_coef": [0.0, 0.01, 0.02],
            "clip_range": [0.2, 0.15, 0.1],
            "net_arch": ["64x64", "128x128", "256x256"]
        }

        # Net arch configurations mappings
        self.net_arch_map = {
            "64x64": {"net_arch": dict(pi=[64, 64], vf=[64, 64])},
            "128x128": {"net_arch": dict(pi=[128, 128], vf=[128, 128])},
            "256x256": {"net_arch": dict(pi=[256, 256], vf=[256, 256])}
        }

    def run_tuning(
        self,
        total_timesteps: int | None = None,
        override_sweep_axes: Dict[str, List[Any]] | None = None,
        override_seeds: List[int] | None = None
    ) -> Dict[str, Any]:
        """Runs the coordinate sensitivity sweep across all axes."""
        if override_seeds is not None:
            self.seeds = override_seeds

        sweep_axes = self.sweep_axes
        if override_sweep_axes is not None:
            sweep_axes = override_sweep_axes

        print(f"\n========================================================")
        print(f"RUNNING HYPERPARAMETER TUNING FOR {self.env_id}")
        print(f"========================================================\n")

        results_summary = {}
        temp_config_dir = os.path.join(self.base_dir, "configs", "temp_tuning")
        os.makedirs(temp_config_dir, exist_ok=True)

        try:
            # 1. Evaluate reference baseline configuration first
            print("Evaluating baseline reference configuration...")
            baseline_name = "tuning/sweep_reference_default"
            baseline_yaml = self._generate_tuning_yaml(
                param_name="reference",
                param_val="default",
                hparam_overrides={},
                temp_dir=temp_config_dir,
                total_timesteps=total_timesteps
            )

            runner = ExperimentRunner(config_path=baseline_yaml, base_dir=self.base_dir)
            runner.run_all()

            ref_stats = self._get_tuning_stats(baseline_name)
            results_summary["reference_default"] = ref_stats

            # 2. Sweep each parameter axis coordinate
            for param, options in sweep_axes.items():
                results_summary[param] = {}
                for opt in options:
                    # If matches baseline value, reuse baseline results and copy directories
                    if opt == self.baseline_hparams[param]:
                        print(f"Option {opt} for {param} matches baseline reference. Skipping execution and copying reference results.")
                        results_summary[param][str(opt)] = ref_stats
                        self._copy_results_to_sweep_folder(baseline_name, f"tuning/sweep_{param}_{opt}")
                        continue

                    print(f"\nTuning: Sweeping parameter {param} = {opt}...")
                    overrides = {}
                    if param == "net_arch":
                        overrides["policy_kwargs"] = self.net_arch_map[opt]
                    else:
                        overrides[param] = opt

                    sweep_name = f"tuning/sweep_{param}_{opt}"
                    sweep_yaml = self._generate_tuning_yaml(
                        param_name=param,
                        param_val=str(opt),
                        hparam_overrides=overrides,
                        temp_dir=temp_config_dir,
                        total_timesteps=total_timesteps
                    )

                    runner = ExperimentRunner(config_path=sweep_yaml, base_dir=self.base_dir)
                    runner.run_all()

                    opt_stats = self._get_tuning_stats(sweep_name)
                    results_summary[param][str(opt)] = opt_stats

        finally:
            if os.path.exists(temp_config_dir):
                shutil.rmtree(temp_config_dir)

        # Save summary JSON
        tuning_results_dir = os.path.join(self.base_dir, "results", self.env_id)
        os.makedirs(tuning_results_dir, exist_ok=True)
        tuning_results_path = os.path.join(tuning_results_dir, "tuning_summary.json")
        with open(tuning_results_path, "w") as f:
            json.dump(results_summary, f, indent=4)

        print(f"\nHyperparameter tuning complete. Results saved: {tuning_results_path}")
        return results_summary

    def _generate_tuning_yaml(
        self,
        param_name: str,
        param_val: str,
        hparam_overrides: Dict[str, Any],
        temp_dir: str,
        total_timesteps: int | None = None
    ) -> str:
        """Generates a temporary configuration YAML file with specified overrides."""
        with open(self.base_config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config_dict["experiment"]["name"] = f"tuning_sweep_{param_name}_{param_val}"
        config_dict["experiment"]["seeds"] = self.seeds
        config_dict["reward_shaping"]["strategy"] = f"tuning/sweep_{param_name}_{param_val}"

        if total_timesteps is not None:
            config_dict["experiment"]["total_timesteps"] = total_timesteps

        # Default config forces CUDA device usage for efficiency
        config_dict["ppo"]["device"] = "cuda"

        # Apply parameter overrides
        for k, v in hparam_overrides.items():
            config_dict["ppo"][k] = v

        temp_file_name = f"tuning_{param_name}_{param_val}.yaml"
        temp_file_path = os.path.join(temp_dir, temp_file_name)
        with open(temp_file_path, "w") as f:
            yaml.dump(config_dict, f)

        return temp_file_path

    def _get_tuning_stats(self, strategy_name: str) -> Dict[str, Any]:
        """Runs ExperimentAnalyzer to extract mean final reward and AUC."""
        from analysis.statistics import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer(env_id=self.env_id, base_dir=self.base_dir)

        summary = analyzer.compute_summary_statistics(strategy_name)

        seed_paths = analyzer._find_seeds_for_strategy(strategy_name)
        max_steps = 100000.0
        if seed_paths:
            meta_path = os.path.join(seed_paths[0], "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    max_steps = float(meta.get("total_timesteps", 100000.0))
                except Exception:
                    pass

        raw_aucs, norm_aucs = analyzer.calculate_auc(strategy_name, max_steps=max_steps)

        mean_reward = summary.get("final_unshaped_reward_mean", np.nan) if summary else np.nan
        std_reward = summary.get("final_unshaped_reward_std", np.nan) if summary else np.nan
        mean_auc = float(np.mean(norm_aucs)) if norm_aucs else np.nan
        std_auc = float(np.std(norm_aucs)) if norm_aucs else np.nan

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "num_seeds": len(seed_paths)
        }

    def _copy_results_to_sweep_folder(self, src_strategy: str, dest_strategy: str) -> None:
        """Copies baseline execution results to the target sweep folder to avoid reduntant execution."""
        for seed in self.seeds:
            seed_suffix = f"seed_{seed}"
            src_res = os.path.join(self.base_dir, "results", self.env_id, src_strategy, seed_suffix)
            dest_res = os.path.join(self.base_dir, "results", self.env_id, dest_strategy, seed_suffix)

            src_logs = os.path.join(self.base_dir, "logs", self.env_id, src_strategy, seed_suffix)
            dest_logs = os.path.join(self.base_dir, "logs", self.env_id, dest_strategy, seed_suffix)

            src_models = os.path.join(self.base_dir, "models", self.env_id, src_strategy, seed_suffix)
            dest_models = os.path.join(self.base_dir, "models", self.env_id, dest_strategy, seed_suffix)

            for src_p, dest_p in [(src_res, dest_res), (src_logs, dest_logs), (src_models, dest_models)]:
                if os.path.exists(src_p):
                    os.makedirs(os.path.dirname(dest_p), exist_ok=True)
                    if os.path.exists(dest_p):
                        shutil.rmtree(dest_p)
                    shutil.copytree(src_p, dest_p)
