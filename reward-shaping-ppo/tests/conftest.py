import json

import numpy as np
import pytest


@pytest.fixture
def synthetic_results_tree(tmp_path):
    env_id = "TestEnv-v0"
    strategies = ["identity", "dense"]
    seeds = [42, 43]
    num_episodes = 50
    num_evals = 10
    eval_episodes = 5

    for strategy in strategies:
        for seed in seeds:
            seed_dir = tmp_path / "results" / env_id / strategy / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            np.random.seed(seed)
            episode_lengths = np.random.randint(20, 200, size=num_episodes)
            if strategy == "dense":
                base_rewards = np.linspace(50, 500, num_episodes) + np.random.normal(
                    0, 20, num_episodes
                )
                shaped_rewards = base_rewards + np.random.uniform(
                    0.5, 2.0, num_episodes
                )
            else:
                base_rewards = np.linspace(30, 500, num_episodes) + np.random.normal(
                    0, 25, num_episodes
                )
                shaped_rewards = base_rewards.copy()

            base_rewards = np.clip(base_rewards, 0, 500)
            shaped_rewards = np.clip(shaped_rewards, 0, 510)

            monitor_path = seed_dir / "monitor.csv"
            with open(monitor_path, "w") as f:
                f.write('#{"t_start": 0.0, "env_id": "TestEnv-v0"}\n')
                f.write("r,l,t,original_reward,shaped_reward\n")
                cumulative_time = 0.0
                for i in range(num_episodes):
                    cumulative_time += episode_lengths[i] * 0.001
                    f.write(
                        f"{base_rewards[i]:.2f},{episode_lengths[i]},{cumulative_time:.4f},"
                        f"{base_rewards[i]:.2f},{shaped_rewards[i]:.2f}\n"
                    )

            eval_monitor_path = seed_dir / "eval_monitor.csv"
            with open(eval_monitor_path, "w") as f:
                f.write('#{"t_start": 0.0, "env_id": "TestEnv-v0"}\n')
                f.write("r,l,t\n")
                for i in range(num_evals):
                    f.write(f"{450 + i * 5:.1f},{200},{i * 10:.1f}\n")

            eval_timesteps = np.linspace(1000, 10000, num_evals).astype(int)
            eval_results = np.array(
                [
                    np.clip(
                        np.full(eval_episodes, 50 + (i * 50))
                        + np.random.normal(0, 5, eval_episodes),
                        0,
                        500,
                    )
                    for i in range(num_evals)
                ]
            )
            eval_lengths = np.full((num_evals, eval_episodes), 200)
            np.savez(
                seed_dir / "evaluations.npz",
                timesteps=eval_timesteps,
                results=eval_results,
                ep_lengths=eval_lengths,
            )

            metadata = {
                "experiment_name": f"test_{strategy}",
                "env_id": env_id,
                "strategy": strategy,
                "seed": seed,
                "total_timesteps": 10000,
                "training_time_seconds": 15.5 + seed * 0.1,
                "device": "cpu",
                "final_model_path": str(seed_dir / "final_model.zip"),
                "deterministic_execution": True,
            }
            with open(seed_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            config_content = f"""experiment:
  name: "test_{strategy}"
  env_id: "TestEnv-v0"
  total_timesteps: 10000
  eval_freq: 1000
  eval_episodes: {eval_episodes}
  checkpoint_freq: 5000
  seeds: [{seed}]
  reproducibility:
    deterministic: true
    benchmark: false

ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  device: "cpu"
  policy_kwargs:
    net_arch:
      pi: [64, 64]
      vf: [64, 64]

reward_shaping:
  strategy: "{strategy}"
  params: {{}}
"""
            with open(seed_dir / "config.yaml", "w") as f:
                f.write(config_content)

    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "plots" / env_id).mkdir(parents=True, exist_ok=True)
    (tmp_path / "paper_assets").mkdir(exist_ok=True)

    return tmp_path, env_id, strategies, seeds


@pytest.fixture
def synthetic_single_strategy_tree(tmp_path):
    env_id = "SimpleEnv-v0"
    strategy = "identity"
    seed = 42

    seed_dir = tmp_path / "results" / env_id / strategy / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    with open(seed_dir / "monitor.csv", "w") as f:
        f.write('#{"t_start": 0.0, "env_id": "SimpleEnv-v0"}\n')
        f.write("r,l,t,original_reward,shaped_reward\n")
        for i in range(20):
            reward = 100 + i * 20
            f.write(f"{reward},{100 + i},{i * 0.1:.2f},{reward},{reward}\n")

    timesteps = np.array([1000, 2000, 3000, 4000, 5000])
    results = np.array(
        [
            [100, 110, 90, 105, 95],
            [200, 210, 190, 205, 195],
            [300, 310, 290, 305, 295],
            [400, 410, 390, 405, 395],
            [500, 500, 500, 500, 500],
        ],
        dtype=float,
    )
    np.savez(
        seed_dir / "evaluations.npz",
        timesteps=timesteps,
        results=results,
        ep_lengths=np.full_like(results, 200),
    )

    metadata = {
        "experiment_name": "test_simple",
        "env_id": env_id,
        "strategy": strategy,
        "seed": seed,
        "total_timesteps": 5000,
        "training_time_seconds": 10.0,
        "device": "cpu",
    }
    with open(seed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return tmp_path, env_id, strategy
