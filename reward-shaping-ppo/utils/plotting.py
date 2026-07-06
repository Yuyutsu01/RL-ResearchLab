import glob
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from analysis.statistics import ExperimentAnalyzer

# Set up matplotlib style settings for publication-grade quality
try:
    plt.style.use("seaborn-v0_8-paper")
except Exception:
    # Fallback to standard style with customizations if seaborn sheet is missing
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.4
    plt.rcParams["grid.linestyle"] = "--"

plt.rcParams["font.serif"] = [
    "Times New Roman",
    "DejaVu Serif",
    "Liberation Serif",
    "serif",
]
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16

# Curated, professional, colorblind-friendly color palette
COLOR_PALETTE = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def save_figure(fig: plt.Figure, base_path: str) -> None:
    """Saves the figure as both high-resolution PNG and vector PDF."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    # Save PNG for web / README
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    # Save PDF for publication/LaTeX documents
    fig.savefig(f"{base_path}.pdf", format="pdf", bbox_inches="tight")
    print(f"Saved figure: {base_path}.png and .pdf")
    plt.close(fig)


def plot_learning_curves(
    env_id: str,
    strategies: list[str],
    base_dir: str = ".",
    grid_points: int = 100,
    rolling_window: int = 10,
) -> None:
    """
    Plots aggregated training learning curves comparing different strategies.
    Plots Mean Original Reward, Mean Shaped Reward, and Episode Lengths with 95% CIs.
    """
    analyzer = ExperimentAnalyzer(env_id, base_dir=base_dir)
    plots_dir = os.path.join(base_dir, "plots", env_id)

    # Load data for all strategies
    strat_data: dict[str, dict[str, Any]] = {}
    for strat in strategies:
        data = analyzer.load_strategy_data(
            strat, grid_points=grid_points, rolling_window=rolling_window
        )
        if data:
            strat_data[strat] = data

    if not strat_data:
        print(f"No training data available to plot for env '{env_id}'")
        return

    # 1. Plot Unshaped (Original) Reward Learning Curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (strat, data) in enumerate(strat_data.items()):
        steps = data["steps"]
        stats = data["original_reward"]
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        ax.plot(
            steps,
            stats["mean"],
            label=f"{strat.capitalize()} (seeds={data['raw_seeds_count']})",
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            steps,
            stats["mean"] - stats["ci95"],
            stats["mean"] + stats["ci95"],
            color=color,
            alpha=0.15,
        )

    ax.set_title(
        f"PPO Training Performance on {env_id} (True Unshaped Objective)", pad=15
    )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Reward (Moving Average)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, os.path.join(plots_dir, "training_original_reward"))

    # 2. Plot Shaped Reward Learning Curves (What the agent actually optimized)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (strat, data) in enumerate(strat_data.items()):
        steps = data["steps"]
        stats = data["shaped_reward"]
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        ax.plot(
            steps,
            stats["mean"],
            label=f"{strat.capitalize()} (Shaped)",
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            steps,
            stats["mean"] - stats["ci95"],
            stats["mean"] + stats["ci95"],
            color=color,
            alpha=0.15,
        )

    ax.set_title(
        f"PPO Training Performance on {env_id} (Proxy Shaped Objective)", pad=15
    )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Shaped Reward (Moving Average)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, os.path.join(plots_dir, "training_shaped_reward"))

    # 3. Plot Episode Lengths
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (strat, data) in enumerate(strat_data.items()):
        steps = data["steps"]
        stats = data["episode_length"]
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        ax.plot(
            steps,
            stats["mean"],
            label=f"{strat.capitalize()}",
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            steps,
            stats["mean"] - stats["ci95"],
            stats["mean"] + stats["ci95"],
            color=color,
            alpha=0.15,
        )

    ax.set_title(f"PPO Episode Lengths during Training on {env_id}", pad=15)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Length")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, os.path.join(plots_dir, "training_episode_lengths"))


def plot_evaluation_curves(
    env_id: str, strategies: list[str], base_dir: str = "."
) -> None:
    """
    Plots evaluation curves from evaluations.npz files generated by EvalCallback.
    Evaluations are computed deterministically on the unshaped environment.
    """
    plots_dir = os.path.join(base_dir, "plots", env_id)
    results_dir = os.path.join(base_dir, "results", env_id)

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False

    for i, strat in enumerate(strategies):
        strat_results_dir = os.path.join(results_dir, strat)
        seed_paths = glob.glob(os.path.join(strat_results_dir, "seed_*"))

        strat_evals = []

        for path in seed_paths:
            npz_path = os.path.join(path, "evaluations.npz")
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                steps = data["timesteps"]
                mean_eval_rew = np.mean(data["results"], axis=1)
                strat_evals.append((steps, mean_eval_rew))

        if len(strat_evals) > 0:
            has_data = True
            # Find the seed with the longest evaluations to use as the alignment grid
            max_idx = int(np.argmax([len(x[0]) for x in strat_evals]))
            target_steps = strat_evals[max_idx][0]

            # Interpolate all seeds onto target_steps
            aligned_evals = []
            for steps, rews in strat_evals:
                if len(steps) == len(target_steps) and np.allclose(steps, target_steps):
                    aligned_evals.append(rews)
                else:
                    interp_rews = np.interp(target_steps, steps, rews)
                    aligned_evals.append(interp_rews)

            eval_steps = target_steps
            eval_arr = np.vstack(aligned_evals)  # shape (num_seeds, n_evaluations)
            mean = np.mean(eval_arr, axis=0)
            sem = (
                stats.sem(eval_arr, axis=0)
                if eval_arr.shape[0] > 1
                else np.zeros_like(mean)
            )
            ci95 = (
                sem * stats.t.ppf((1 + 0.95) / 2.0, eval_arr.shape[0] - 1)
                if eval_arr.shape[0] > 1
                else np.zeros_like(mean)
            )
            ci95 = np.nan_to_num(ci95)

            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            ax.plot(
                eval_steps,
                mean,
                label=f"{strat.capitalize()} (seeds={eval_arr.shape[0]})",
                color=color,
                linewidth=2,
            )
            ax.fill_between(
                eval_steps, mean - ci95, mean + ci95, color=color, alpha=0.15
            )

    if not has_data:
        print(f"No evaluation NPZ data files found for env '{env_id}'")
        plt.close(fig)
        return

    ax.set_title(
        f"PPO Evaluation Performance on {env_id} (Deterministic, Unshaped)", pad=15
    )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Evaluation Mean Reward")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, os.path.join(plots_dir, "evaluation_curves"))


def parse_tensorboard_loss(log_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Parses TensorBoard event logs to extract training scalar metrics.

    Args:
        log_dir: Directory containing tfevents files.

    Returns:
        A dictionary mapping metric name -> tuple of (steps, values) arrays.
    """
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return {}

    try:
        ea = EventAccumulator(event_files[0])
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        extracted = {}

        targets = [
            "train/loss",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/learning_rate",
        ]

        for tag in targets:
            if tag in tags:
                events = ea.Scalars(tag)
                steps = np.array([e.step for e in events])
                values = np.array([e.value for e in events])
                extracted[tag] = (steps, values)

        return extracted
    except Exception as e:
        print(f"Failed to parse TensorBoard logs in {log_dir}: {e}")
        return {}


def plot_training_losses(
    env_id: str, strategy: str, seed: int, base_dir: str = "."
) -> None:
    """
    Plots the policy gradient loss, value loss, total loss, and policy entropy
    over training steps for a single seed.
    """
    seed_suffix = f"seed_{seed}"
    log_dir = os.path.join(base_dir, "logs", env_id, strategy, seed_suffix)
    plots_dir = os.path.join(base_dir, "plots", env_id, strategy, seed_suffix)

    # We look inside the PPO_* subfolders created by SB3
    subdirs = glob.glob(os.path.join(log_dir, "PPO_*"))
    target_log_dir = subdirs[0] if subdirs else log_dir

    metrics = parse_tensorboard_loss(target_log_dir)
    if not metrics:
        print(f"No Tensorboard metrics parsed from {target_log_dir}")
        return

    # We will generate a 2x2 grid plot of the loss metrics
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"PPO Training Losses: {env_id} | {strategy.capitalize()} (Seed {seed})", y=0.96
    )

    # Map layout:
    # 0,0: Total Loss       | 0,1: Value Loss
    # 1,0: Policy Loss      | 1,1: Policy Entropy

    mapping = [
        ("train/loss", axs[0, 0], "Total Training Loss", "#d95f02"),
        ("train/value_loss", axs[0, 1], "Value Function Loss", "#7570b3"),
        ("train/policy_gradient_loss", axs[1, 0], "Policy Gradient Loss", "#1b9e77"),
        ("train/entropy_loss", axs[1, 1], "Policy Entropy Loss", "#e7298a"),
    ]

    for tag, ax, title, color in mapping:
        if tag in metrics:
            steps, vals = metrics[tag]
            ax.plot(steps, vals, color=color, linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel("Value")
        else:
            ax.text(0.5, 0.5, "Metric Not Found", dict(ha="center", va="center"))

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    save_figure(fig, os.path.join(plots_dir, "losses"))
