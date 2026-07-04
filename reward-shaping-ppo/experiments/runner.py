import json
import shutil
from typing import Dict, Any
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback,
)


from callbacks.logging_callback import ResearchLoggingCallback
from environments.wrapper import RewardShapingWrapper
from reward_functions import get_reward_shaper
from utils.config import Config
from utils.reproducibility import set_seed



class ExperimentRunner:
    """
    Orchestrates reinforcement learning experiments, setting up environments,
    configuring agents, executing training across multiple seeds, and exporting logs.
    """

    def __init__(self, config_path: str, base_dir: str = "."):
        """
        Initializes the ExperimentRunner.

        Args:
            config_path: Path to the YAML configuration file.
            base_dir: Root directory of the reward-shaping framework.
        """
        self.config_path = config_path
        self.base_dir = base_dir
        self.config = Config.from_yaml(config_path)

        # Output directory templates
        self.env_id = self.config.experiment.env_id
        self.strategy = self.config.reward_shaping.strategy
        self.experiment_name = self.config.experiment.name

    def _create_directories(self, seed: int) -> Dict[str, str]:
        """
        Creates clean execution subdirectories for a specific seed.

        Returns:
            A dictionary containing paths for models, logs, and results.
        """
        seed_suffix = f"seed_{seed}"

        paths = {
            "model_dir": os.path.abspath(
                os.path.join(
                    self.base_dir, "models", self.env_id, self.strategy, seed_suffix
                )
            ),
            "log_dir": os.path.abspath(
                os.path.join(
                    self.base_dir, "logs", self.env_id, self.strategy, seed_suffix
                )
            ),
            "result_dir": os.path.abspath(
                os.path.join(
                    self.base_dir, "results", self.env_id, self.strategy, seed_suffix
                )
            ),
        }

        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        return paths

    def run_single_seed(self, seed: int) -> dict[str, Any]:
        """
        Runs the experiment for a single seed.

        Args:
            seed: The random seed to evaluate.

        Returns:
            A summary dictionary containing experiment diagnostics.
        """
        print("\n========================================================")
        print(
            f"Starting Seed {seed} | Environment: {self.env_id} | Strategy: {self.strategy}"
        )
        print("========================================================\n")

        # Set up reproducibility
        set_seed(
            seed=seed,
            deterministic=self.config.experiment.reproducibility.deterministic,
            benchmark=self.config.experiment.reproducibility.benchmark,
        )

        # Setup folders
        paths = self._create_directories(seed)

        # Save experiment config locally to results for validation audit trail
        shutil.copy2(self.config_path, os.path.join(paths["result_dir"], "config.yaml"))

        # 1. Initialize environments
        # Training Environment: wrapped in shaping + monitor
        def make_train_env():
            raw_env = gym.make(self.env_id)
            raw_env.action_space.seed(seed)
            raw_env.observation_space.seed(seed)

            shaper = get_reward_shaper(self.strategy, self.config.reward_shaping.params)
            shaped_env = RewardShapingWrapper(raw_env, shaper)

            # Monitor logs to CSV
            monitor_csv = os.path.join(paths["result_dir"], "monitor.csv")
            monitored_env = Monitor(
                shaped_env,
                filename=monitor_csv,
                info_keywords=("original_reward", "shaped_reward"),
            )
            return monitored_env

        # Evaluation Environment: wrapped in raw/identity wrapper (unbiased target evaluation)
        def make_eval_env():
            raw_env = gym.make(self.env_id)
            raw_env.action_space.seed(
                seed + 100
            )  # Offset seed to prevent initial state correlation
            raw_env.observation_space.seed(seed + 100)

            identity_shaper = get_reward_shaper("identity")
            eval_env = RewardShapingWrapper(raw_env, identity_shaper)

            eval_csv = os.path.join(paths["result_dir"], "eval_monitor.csv")
            monitored_eval_env = Monitor(
                eval_env,
                filename=eval_csv,
                info_keywords=("original_reward", "shaped_reward"),
            )
            return monitored_eval_env

        train_env = DummyVecEnv([make_train_env])
        eval_env = DummyVecEnv([make_eval_env])

        # 2. Build Callback List
        logging_callback = ResearchLoggingCallback()

        # Periodic model checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.experiment.checkpoint_freq,
            save_path=paths["model_dir"],
            name_prefix="ppo_checkpoint",
            verbose=0,
        )

        # Periodic evaluation tracking (on unbiased unshaped environment)
        eval_callback = EvalCallback(
            eval_env=eval_env,
            callback_on_new_best=None,
            n_eval_episodes=self.config.experiment.eval_episodes,
            eval_freq=self.config.experiment.eval_freq,
            log_path=paths["result_dir"],
            best_model_save_path=paths["model_dir"],
            deterministic=True,
            verbose=1,
        )

        callbacks = CallbackList([logging_callback, checkpoint_callback, eval_callback])

        # 3. Initialize stable-baselines3 PPO
        device = self.config.ppo.device
        print(f"Executing on device: {device}")

        ppo_hparams = self.config.ppo
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=ppo_hparams.learning_rate,
            n_steps=ppo_hparams.n_steps,
            batch_size=ppo_hparams.batch_size,
            n_epochs=ppo_hparams.n_epochs,
            gamma=ppo_hparams.gamma,
            gae_lambda=ppo_hparams.gae_lambda,
            clip_range=ppo_hparams.clip_range,
            ent_coef=ppo_hparams.ent_coef,
            vf_coef=ppo_hparams.vf_coef,
            max_grad_norm=ppo_hparams.max_grad_norm,
            policy_kwargs=ppo_hparams.policy_kwargs,
            tensorboard_log=paths["log_dir"],
            seed=seed,
            device=device,
            verbose=0,
        )

        # 4. Train Agent
        start_time = time.time()
        model.learn(
            total_timesteps=self.config.experiment.total_timesteps,
            callback=callbacks,
            tb_log_name="PPO",
        )
        training_time = time.time() - start_time

        # 5. Save Final Model
        final_model_path = os.path.join(paths["model_dir"], "final_model.zip")
        model.save(final_model_path)

        # Close environments
        train_env.close()
        eval_env.close()

        # 6. Save Metadata summary
        summary = {
            "experiment_name": self.experiment_name,
            "env_id": self.env_id,
            "strategy": self.strategy,
            "seed": seed,
            "total_timesteps": self.config.experiment.total_timesteps,
            "training_time_seconds": training_time,
            "device": str(device),
            "final_model_path": final_model_path,
            "deterministic_execution": self.config.experiment.reproducibility.deterministic,
        }

        with open(os.path.join(paths["result_dir"], "metadata.json"), "w") as f:
            json.dump(summary, f, indent=4)

        print(f"Seed {seed} finished successfully in {training_time:.2f} seconds.")
        return summary

    def run_all(self) -> dict[int, dict[str, Any]]:
        """
        Runs the experiment sequentially for all seeds defined in the config.

        Returns:
            A dictionary mapping seed -> execution diagnostic summary.
        """
        results = {}
        seeds = self.config.experiment.seeds

        print(f"Starting experiment suite: '{self.experiment_name}'")
        print(f"Environments: {self.env_id} | Strategy: {self.strategy}")
        print(f"Running across seeds: {seeds}")

        suite_start_time = time.time()

        for seed in seeds:
            results[seed] = self.run_single_seed(seed)

        suite_duration = time.time() - suite_start_time
        print("\n========================================================")
        print(f"All experiments in suite completed! Total time: {suite_duration:.2f}s")
        print("========================================================\n")

        return results
