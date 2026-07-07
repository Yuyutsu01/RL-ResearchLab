import datetime
import glob
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class ExperimentAnalyzer:
    """
    Analyzes monitor logs across multiple seeds for reinforcement learning experiments.
    Computes aggregated statistics, rolling averages, confidence intervals,
    and formats data for scientific plotting and comparison.
    """

    def __init__(self, env_id: str, base_dir: str = "."):
        """
        Initializes the ExperimentAnalyzer.

        Args:
            env_id: Gymnasium environment ID (e.g., 'CartPole-v1').
            base_dir: Root directory of the framework.
        """
        self.env_id = env_id
        self.base_dir = os.path.abspath(base_dir)
        self.results_dir = os.path.join(self.base_dir, "results", env_id)

    def _find_seeds_for_strategy(self, strategy: str) -> list[str]:
        """Finds all seed directory paths for a given reward shaping strategy."""
        strategy_dir = os.path.join(self.results_dir, strategy)
        if not os.path.exists(strategy_dir):
            return []
        # Find paths matching seed_*
        seed_paths = glob.glob(os.path.join(strategy_dir, "seed_*"))
        return [p for p in seed_paths if os.path.isdir(p)]

    def load_strategy_data(
        self, strategy: str, grid_points: int = 100, rolling_window: int = 10
    ) -> dict[str, Any] | None:
        """
        Loads, cleans, and aggregates monitor logs across all seeds for a strategy.
        Interpolates data onto a common grid of step counts to allow cross-seed aggregation.

        Args:
            strategy: Reward shaping strategy name.
            grid_points: Number of points to interpolate onto.
            rolling_window: Moving average window for raw episode rewards.

        Returns:
            A dictionary containing aggregated grid statistics, or None if no data found.
        """
        seed_paths = self._find_seeds_for_strategy(strategy)
        if not seed_paths:
            print(
                f"No log directories found for strategy '{strategy}' in env '{self.env_id}'"
            )
            return None

        all_original_rewards = []
        all_shaped_rewards = []
        all_episode_lengths = []
        max_steps = 0

        # Load raw data from each seed
        seed_dfs = []
        for path in seed_paths:
            csv_path = os.path.join(path, "monitor.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                # SB3 monitor files have a comment in the first line, column header in the second
                df = pd.read_csv(csv_path, skiprows=1)
                if len(df) == 0:
                    continue
                # Calculate cumulative training steps at each episode termination
                df["cumulative_steps"] = df["l"].cumsum()
                max_steps = max(max_steps, df["cumulative_steps"].max())
                seed_dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")

        if not seed_dfs:
            return None

        # Define a common steps grid for alignment
        grid_steps = np.linspace(0, max_steps, grid_points)

        for df in seed_dfs:
            # Apply rolling average to smooth individual runs before interpolation
            rolling_orig = (
                df["original_reward"]
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )
            rolling_shape = (
                df["shaped_reward"]
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )
            rolling_len = (
                df["l"].rolling(window=rolling_window, min_periods=1).mean().values
            )

            # Interpolate onto common step grid
            interp_orig = np.interp(
                grid_steps,
                df["cumulative_steps"].values,
                rolling_orig,
                left=rolling_orig[0],
                right=rolling_orig[-1],
            )
            interp_shape = np.interp(
                grid_steps,
                df["cumulative_steps"].values,
                rolling_shape,
                left=rolling_shape[0],
                right=rolling_shape[-1],
            )
            interp_len = np.interp(
                grid_steps,
                df["cumulative_steps"].values,
                rolling_len,
                left=rolling_len[0],
                right=rolling_len[-1],
            )

            all_original_rewards.append(interp_orig)
            all_shaped_rewards.append(interp_shape)
            all_episode_lengths.append(interp_len)

        # Stack arrays: (num_seeds, grid_points)
        orig_arr = np.vstack(all_original_rewards)
        shape_arr = np.vstack(all_shaped_rewards)
        len_arr = np.vstack(all_episode_lengths)

        # Calculate statistics per grid point
        def compute_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            median = np.median(arr, axis=0)
            sem = stats.sem(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
            # 95% Confidence Interval half-width
            ci = (
                sem * stats.t.ppf((1 + 0.95) / 2.0, arr.shape[0] - 1)
                if arr.shape[0] > 1
                else np.zeros_like(mean)
            )
            # Handle NaN values from single seed statistics
            ci = np.nan_to_num(ci)

            return {
                "mean": mean,
                "std": std,
                "median": median,
                "sem": sem,
                "ci95": ci,
                "min": np.min(arr, axis=0),
                "max": np.max(arr, axis=0),
            }

        return {
            "steps": grid_steps,
            "original_reward": compute_stats(orig_arr),
            "shaped_reward": compute_stats(shape_arr),
            "episode_length": compute_stats(len_arr),
            "raw_seeds_count": len(seed_dfs),
        }

    def compute_summary_statistics(
        self, strategy: str, last_pct: float = 0.10, step_limit: float | None = None
    ) -> dict[str, Any] | None:
        """
        Computes summary statistics, optionally up to step_limit.
        """
        seed_paths = self._find_seeds_for_strategy(strategy)
        if not seed_paths:
            return None

        final_orig_rewards = []
        training_times = []

        for path in seed_paths:
            # Load monitor data
            csv_path = os.path.join(path, "monitor.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, skiprows=1)
                    if len(df) > 0:
                        df["cumulative_steps"] = df["l"].cumsum()
                        if step_limit is not None:
                            df = df[df["cumulative_steps"] <= step_limit]
                        if len(df) > 0:
                            # Determine starting index for final window
                            num_episodes = len(df)
                            start_idx = int(num_episodes * (1.0 - last_pct))
                            final_orig_rewards.append(
                                df["original_reward"].iloc[start_idx:].mean()
                            )
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")

            # Load metadata
            meta_path = os.path.join(path, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    total_t = meta.get("total_timesteps", 1.0)
                    raw_time = meta.get("training_time_seconds", 0.0)
                    if step_limit is not None and total_t > 0:
                        # Estimate time linearly up to step_limit
                        t_ratio = min(1.0, step_limit / total_t)
                        training_times.append(raw_time * t_ratio)
                    else:
                        training_times.append(raw_time)
                except Exception as e:
                    print(f"Error loading {meta_path}: {e}")

        if not final_orig_rewards:
            return None

        orig_arr = np.array(final_orig_rewards)
        time_arr = np.array(training_times) if training_times else np.array([0.0])

        mean_orig = float(np.mean(orig_arr))
        std_orig = float(np.std(orig_arr))
        sem_orig = float(stats.sem(orig_arr)) if len(orig_arr) > 1 else 0.0
        ci_orig = (
            float(sem_orig * stats.t.ppf((1 + 0.95) / 2.0, len(orig_arr) - 1))
            if len(orig_arr) > 1
            else 0.0
        )

        summary = {
            "strategy": strategy,
            "num_seeds": len(final_orig_rewards),
            "final_unshaped_reward_mean": mean_orig,
            "final_unshaped_reward_std": std_orig,
            "final_unshaped_reward_ci95": ci_orig,
            "mean_training_time_seconds": float(np.mean(time_arr)),
            "total_training_time_seconds": float(np.sum(time_arr)),
        }

        # Save summary to strategy folder only if step_limit is None (main summary)
        if step_limit is None:
            summary_path = os.path.join(self.results_dir, strategy, "summary.json")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)

        return summary

    def _get_default_thresholds(self) -> list[float]:
        """Returns standard evaluation thresholds for the environment."""
        if "MountainCar" in self.env_id:
            return [-180.0, -160.0, -140.0, -120.0, -110.0]
        elif "Acrobot" in self.env_id:
            return [-300.0, -200.0, -150.0, -100.0, -80.0]
        elif "LunarLander" in self.env_id:
            return [-100.0, 0.0, 50.0, 100.0, 200.0]
        else: # CartPole and defaults
            return [100.0, 200.0, 300.0, 400.0, 500.0]

    def _get_key_threshold(self) -> float:
        """Returns the primary target threshold for the environment."""
        if "MountainCar" in self.env_id:
            return -120.0
        elif "Acrobot" in self.env_id:
            return -100.0
        elif "LunarLander" in self.env_id:
            return 100.0
        else: # CartPole and defaults
            return 400.0

    def calculate_auc(self, strategy: str, max_steps: float) -> tuple[list[float], list[float]]:
        """
        Computes raw and normalized AUC per seed up to max_steps using evaluations.npz.
        Returns (raw_aucs, normalized_aucs) lists.
        """
        seed_paths = self._find_seeds_for_strategy(strategy)
        raw_aucs = []
        norm_aucs = []
        for path in seed_paths:
            npz_path = os.path.join(path, "evaluations.npz")
            if os.path.exists(npz_path):
                try:
                    data = np.load(npz_path)
                    timesteps = data["timesteps"]
                    results = data["results"]
                    mean_rewards = np.mean(results, axis=1)

                    # Filter to max_steps
                    mask = timesteps <= max_steps
                    s_steps = list(timesteps[mask])
                    s_rews = list(mean_rewards[:len(s_steps)])

                    if not s_steps:
                        raw_aucs.append(0.0)
                        norm_aucs.append(0.0)
                        continue

                    # Prepend 0 if not present
                    if s_steps[0] > 0:
                        s_steps.insert(0, 0.0)
                        s_rews.insert(0, s_rews[0])

                    # Append max_steps if last step is less
                    if s_steps[-1] < max_steps:
                        s_steps.append(max_steps)
                        s_rews.append(s_rews[-1])

                    # Integrate using numpy's trapezoidal method compatible with numpy 2.0+
                    if hasattr(np, "trapezoid"):
                        raw_auc = np.trapezoid(s_rews, s_steps)
                    else:
                        raw_auc = np.trapz(s_rews, s_steps)

                    norm_auc = raw_auc / max_steps if max_steps > 0 else 0.0
                    raw_aucs.append(float(raw_auc))
                    norm_aucs.append(float(norm_auc))
                except Exception as e:
                    print(f"Error calculating AUC in {npz_path}: {e}")
                    raw_aucs.append(0.0)
                    norm_aucs.append(0.0)
            else:
                raw_aucs.append(0.0)
                norm_aucs.append(0.0)
        return raw_aucs, norm_aucs

    def calculate_cliffs_delta(self, group1: list[float], group2: list[float]) -> float:
        """
        Computes Cliff's Delta non-parametric effect size between group1 and group2.
        """
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0

        greater = 0
        less = 0
        for x in group1:
            for y in group2:
                if x > y:
                    greater += 1
                elif x < y:
                    less += 1
        return (greater - less) / (n1 * n2)

    def calculate_bootstrap_ci(
        self, data: list[float], confidence: float = 0.95, num_resamples: int = 10000
    ) -> tuple[float, float]:
        """
        Computes bootstrap percentile confidence interval for the mean of the data.
        """
        n = len(data)
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return float(data[0]), float(data[0])

        arr = np.array(data)
        if np.all(arr == arr[0]):
            return float(arr[0]), float(arr[0])

        np.random.seed(42)  # For reproducibility
        boot_means = []
        for _ in range(num_resamples):
            sample = np.random.choice(arr, size=n, replace=True)
            boot_means.append(np.mean(sample))

        lower_pct = (1.0 - confidence) / 2.0 * 100.0
        upper_pct = (1.0 + confidence) / 2.0 * 100.0

        lower = np.percentile(boot_means, lower_pct)
        upper = np.percentile(boot_means, upper_pct)
        return float(lower), float(upper)

    def apply_benjamini_hochberg(self, p_values: list[float], alpha: float = 0.05) -> list[bool]:
        """
        Applies Benjamini-Hochberg False Discovery Rate correction to p-values.
        Returns a list of boolean flags indicating significance.
        """
        m = len(p_values)
        if m == 0:
            return []

        indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
        reject = [False] * m
        max_k = -1
        for rank, (orig_idx, p_val) in enumerate(indexed_p):
            k = rank + 1
            if p_val <= (k / m) * alpha:
                max_k = rank

        if max_k >= 0:
            for rank in range(max_k + 1):
                orig_idx = indexed_p[rank][0]
                reject[orig_idx] = True

        return reject

    def get_timesteps_to_fractional_thresholds(
        self, strategy: str, fractions: list[float] | None = None
    ) -> dict[float, dict[str, Any]]:
        """
        Calculates average timesteps required to reach fractional thresholds of optimal performance span.
        """
        if fractions is None:
            fractions = [0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 1.00]

        if "MountainCar" in self.env_id:
            r_min, r_max = -200.0, -110.0
        elif "Acrobot" in self.env_id:
            r_min, r_max = -500.0, -80.0
        elif "LunarLander" in self.env_id:
            r_min, r_max = -250.0, 200.0
        else: # CartPole and defaults
            r_min, r_max = 10.0, 500.0

        thresholds = [r_min + f * (r_max - r_min) for f in fractions]
        raw_results = self.get_timesteps_to_thresholds(strategy, thresholds=thresholds)

        fractional_results = {}
        for f, t in zip(fractions, thresholds):
            if t in raw_results:
                fractional_results[f] = raw_results[t]
            else:
                closest_key = min(raw_results.keys(), key=lambda k: abs(k - t))
                fractional_results[f] = raw_results[closest_key]

        return fractional_results

    def get_timesteps_to_thresholds(
        self, strategy: str, thresholds: list[float] | list[int] | None = None
    ) -> dict[float, dict[str, Any]]:
        """
        Calculates the average timesteps required to reach specific evaluation
        reward thresholds for a given strategy.
        """
        if thresholds is None:
            thresholds = self._get_default_thresholds()

        seed_paths = self._find_seeds_for_strategy(strategy)
        if not seed_paths:
            return {}

        threshold_steps = {float(t): [] for t in thresholds}

        for path in seed_paths:
            npz_path = os.path.join(path, "evaluations.npz")
            if os.path.exists(npz_path):
                try:
                    data = np.load(npz_path)
                    timesteps = data["timesteps"]
                    results = data["results"]  # Shape: (n_evaluations, n_eval_episodes)
                    mean_rewards = np.mean(results, axis=1)

                    for t in thresholds:
                        reached_idx = np.where(mean_rewards >= t)[0]
                        if len(reached_idx) > 0:
                            step = float(timesteps[reached_idx[0]])
                            threshold_steps[float(t)].append(step)
                        else:
                            threshold_steps[float(t)].append(np.nan)
                except Exception as e:
                    print(f"Error reading evaluations in {npz_path}: {e}")

        results_dict = {}
        for t in thresholds:
            vals = np.array(threshold_steps[float(t)])
            valid_vals = vals[~np.isnan(vals)]

            mean_val = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan
            std_val = float(np.std(valid_vals)) if len(valid_vals) > 0 else np.nan

            results_dict[float(t)] = {
                "mean": mean_val,
                "std": std_val,
                "values": list(vals),
            }

        return results_dict

    def get_final_evaluation_rewards(
        self, strategy: str, step_limit: float | None = None
    ) -> list[float]:
        """Retrieves the final unshaped evaluation score across seeds for a strategy, optionally sliced at step_limit."""
        seed_paths = self._find_seeds_for_strategy(strategy)
        rewards = []
        for path in seed_paths:
            npz_path = os.path.join(path, "evaluations.npz")
            if os.path.exists(npz_path):
                try:
                    data = np.load(npz_path)
                    timesteps = data["timesteps"]
                    results = data["results"]
                    if step_limit is not None:
                        mask = timesteps <= step_limit
                        if np.sum(mask) > 0:
                            idx = np.where(mask)[0][-1]
                            rewards.append(float(np.mean(results[idx])))
                        else:
                            rewards.append(float(np.mean(results[0])))
                    else:
                        rewards.append(float(np.mean(results[-1])))
                except Exception as e:
                    print(f"Error loading final eval in {npz_path}: {e}")
        return rewards

    def perform_statistical_tests(self, strat1: str, strat2: str) -> dict[str, Any]:
        """
        Performs independent t-test (Welch's), Mann-Whitney U test, and Cohens d
        effect size calculations comparing final evaluation scores and steps to thresholds.
        """
        rewards1 = self.get_final_evaluation_rewards(strat1)
        rewards2 = self.get_final_evaluation_rewards(strat2)

        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            if n1 == 0 or n2 == 0:
                return 0.0

            # If a group has only 1 element, use ddof=0 to avoid division by zero (variance is 0)
            ddof1 = 1 if n1 > 1 else 0
            ddof2 = 1 if n2 > 1 else 0

            var1, var2 = np.var(group1, ddof=ddof1), np.var(group2, ddof=ddof2)
            mean1, mean2 = np.mean(group1), np.mean(group2)

            denominator = ((n1 - 1) * var1 if n1 > 1 else 0.0) + (
                (n2 - 1) * var2 if n2 > 1 else 0.0
            )
            divisor = n1 + n2 - 2

            if divisor <= 0:
                # Fallback standard deviation across all combined values
                all_vals = list(group1) + list(group2)
                std_all = np.std(all_vals)
                if std_all == 0:
                    return 0.0
                return (mean1 - mean2) / std_all

            pooled_std = np.sqrt(denominator / divisor)
            if pooled_std == 0 or np.isnan(pooled_std):
                all_vals = list(group1) + list(group2)
                std_all = np.std(all_vals)
                if std_all == 0:
                    return 0.0
                return (mean1 - mean2) / std_all

            return (mean1 - mean2) / pooled_std

        # Run tests on final rewards
        t_stat, t_p = (
            stats.ttest_ind(rewards1, rewards2, equal_var=False)
            if len(rewards1) > 1 and len(rewards2) > 1
            else (0.0, 1.0)
        )
        u_stat, u_p = (
            stats.mannwhitneyu(rewards1, rewards2, alternative="two-sided")
            if len(rewards1) > 0 and len(rewards2) > 0
            else (0.0, 1.0)
        )
        d_val = cohens_d(rewards1, rewards2)

        # Run tests on thresholds
        thresh1 = self.get_timesteps_to_thresholds(strat1)
        thresh2 = self.get_timesteps_to_thresholds(strat2)
        thresholds_comparison = {}

        for t in self._get_default_thresholds():
            if t in thresh1 and t in thresh2:
                v1 = np.array(thresh1[t]["values"])
                v2 = np.array(thresh2[t]["values"])

                # Filter out nans
                v1_clean = v1[~np.isnan(v1)]
                v2_clean = v2[~np.isnan(v2)]

                if len(v1_clean) > 1 and len(v2_clean) > 1:
                    t_stat_t, t_p_t = stats.ttest_ind(
                        v1_clean, v2_clean, equal_var=False
                    )
                    u_stat_t, u_p_t = stats.mannwhitneyu(
                        v1_clean, v2_clean, alternative="two-sided"
                    )
                    d_val_t = cohens_d(v1_clean, v2_clean)
                else:
                    t_stat_t, t_p_t, u_stat_t, u_p_t, d_val_t = 0.0, 1.0, 0.0, 1.0, 0.0

                thresholds_comparison[t] = {
                    f"{strat1}_mean": thresh1[t]["mean"],
                    f"{strat2}_mean": thresh2[t]["mean"],
                    "t_statistic": float(t_stat_t),
                    "t_p_value": float(t_p_t),
                    "u_statistic": float(u_stat_t),
                    "u_p_value": float(u_p_t),
                    "cohens_d": float(d_val_t),
                }

        return {
            "final_rewards": {
                f"{strat1}_values": rewards1,
                f"{strat2}_values": rewards2,
                "t_statistic": float(t_stat),
                "t_p_value": float(t_p),
                "u_statistic": float(u_stat),
                "u_p_value": float(u_p),
                "cohens_d": float(d_val),
            },
            "thresholds": thresholds_comparison,
        }

    def generate_statistical_report(self, strat1: str, strat2: str) -> None:
        """
        Runs statistical tests, structures tables, compiles an ASCII report,
        and saves all comparative assets to the paper_assets/ directory.
        """
        paper_assets_dir = os.path.join(self.base_dir, "paper_assets")
        os.makedirs(paper_assets_dir, exist_ok=True)

        # 1. Run tests
        results = self.perform_statistical_tests(strat1, strat2)

        # Save JSON data
        json_path = os.path.join(
            paper_assets_dir, f"statistical_tests_{strat1}_vs_{strat2}.json"
        )
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        # 2. Compile Text Report
        fr = results["final_rewards"]
        tr = results["thresholds"]

        report = f"""================================================================================
RL RESEARCH LAB: STATISTICAL EVALUATION MANUSCRIPT ASSETS
Comparing: {strat1.upper()} (Control) vs. {strat2.upper()} (Shaped)
Generated: {datetime.date.today().strftime("%Y-%m-%d")}
================================================================================

1. FINAL DETErMINISTIC EVALUATION PERFORMANCE
--------------------------------------------------------------------------------
- {strat1.upper()} raw rewards: {fr[f'{strat1}_values']}
- {strat2.upper()} raw rewards: {fr[f'{strat2}_values']}

Welch's Independent t-test:
  t-statistic = {fr['t_statistic']:.4f}
  p-value     = {fr['t_p_value']:.4e} (p < 0.05 is statistically significant)

Mann-Whitney U Rank Test:
  U-statistic = {fr['u_statistic']:.4f}
  p-value     = {fr['u_p_value']:.4e}

Effect Size:
  Cohen's d   = {fr['cohens_d']:.4f} (d > 0.8 represents large effect size)

Conclusion:
  {"The strategies differ significantly in final asymptotic performance." if fr['t_p_value'] < 0.05 else "No statistically significant difference in final performance was detected."}

2. SAMPLE EFFICIENCY AND CONVERGENCE DYNAMICS
--------------------------------------------------------------------------------
Average timesteps required to reach unshaped evaluation reward thresholds:

| Threshold | {strat1.capitalize()} Mean (Steps) | {strat2.capitalize()} Mean (Steps) | Speedup Factor | t-p-value | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""

        csv_rows = []
        for t in self._get_default_thresholds():
            t_data = tr.get(t, {})
            m1 = t_data.get(f"{strat1}_mean", np.nan)
            m2 = t_data.get(f"{strat2}_mean", np.nan)

            speedup = (
                m1 / m2
                if m1 > 0 and m2 > 0 and not np.isnan(m1) and not np.isnan(m2)
                else np.nan
            )
            p_val = t_data.get("t_p_value", 1.0)
            d_val = t_data.get("cohens_d", 0.0)

            m1_str = f"{m1:.1f}" if not np.isnan(m1) else "N/A"
            m2_str = f"{m2:.1f}" if not np.isnan(m2) else "N/A"
            speedup_str = f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A"

            report += f"| Reward {t:6.1f} | {m1_str:21s} | {m2_str:21s} | {speedup_str:14s} | {p_val:.4e} | {d_val:.4f} |\n"

            csv_rows.append(
                {
                    "Threshold": t,
                    f"{strat1}_Mean_Steps": m1,
                    f"{strat2}_Mean_Steps": m2,
                    "Speedup": speedup,
                    "t_p_value": p_val,
                    "cohens_d": d_val,
                }
            )

        report += "\n================================================================================"

        # Save text report
        txt_path = os.path.join(paper_assets_dir, "statistical_summary.txt")
        with open(txt_path, "w") as f:
            f.write(report)
        print(f"Generated text report: {txt_path}")

        # Save CSV comparison table
        csv_path = os.path.join(paper_assets_dir, "comparison_table.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Generated CSV table: {csv_path}")

    def generate_comparison_report(self, strategies: list[str]) -> pd.DataFrame:
        """
        Aggregates summary statistics for all provided strategies and
        generates a Markdown-compatible Pandas DataFrame comparison table.
        """
        rows = []
        key_thresh = self._get_key_threshold()
        for strat in strategies:
            summary = self.compute_summary_statistics(strat)
            if summary:
                thresh_data = self.get_timesteps_to_thresholds(strat)
                # Steps to reach primary threshold
                steps_key = thresh_data.get(key_thresh, {}).get("mean", np.nan)
                # Float fallback
                if np.isnan(steps_key):
                    closest_key = min(thresh_data.keys(), key=lambda k: abs(k - key_thresh))
                    steps_key = thresh_data.get(closest_key, {}).get("mean", np.nan)
                steps_key_str = f"{steps_key:.1f}" if not np.isnan(steps_key) else "N/A"

                rows.append(
                    {
                        "Strategy": strat.capitalize(),
                        "Seeds": summary["num_seeds"],
                        "Final Reward (Mean ± SD)": f"{summary['final_unshaped_reward_mean']:.2f} ± {summary['final_unshaped_reward_std']:.2f}",
                        "95% CI": f"± {summary['final_unshaped_reward_ci95']:.2f}",
                        f"Steps to R>={key_thresh}": steps_key_str,
                        "Mean Train Time (s)": f"{summary['mean_training_time_seconds']:.1f}s",
                    }
                )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.results_dir, "comparison_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nComparison report saved to {csv_path}")
        return df

    def generate_latex_and_manuscript_assets(self, strategies: list[str]) -> None:
        """
        Compiles unified multi-strategy comparison tables, pairwise statistical tests,
        and LaTeX code for inclusion in the research manuscript.
        """
        import shutil

        paper_assets_dir = os.path.join(self.base_dir, "paper_assets")
        docs_paper_dir = os.path.join(self.base_dir, "docs", "paper")
        os.makedirs(paper_assets_dir, exist_ok=True)
        os.makedirs(docs_paper_dir, exist_ok=True)

        # 1. Compile Unified CSV Comparison Table
        unified_rows = []
        for strat in strategies:
            summary = self.compute_summary_statistics(strat)
            if not summary:
                continue
            thresh_data = self.get_timesteps_to_thresholds(strat)

            row = {
                "Strategy": strat.capitalize(),
                "Seeds": int(summary["num_seeds"]),
                "Final Reward (Mean)": float(summary["final_unshaped_reward_mean"]),
                "Final Reward (Std)": float(summary["final_unshaped_reward_std"]),
                "95% CI": float(summary["final_unshaped_reward_ci95"]),
                "Mean Train Time (s)": float(summary["mean_training_time_seconds"]),
            }
            # Add steps to thresholds
            for t in self._get_default_thresholds():
                val = thresh_data.get(t, {}).get("mean", np.nan)
                row[f"Steps to R>={t}"] = float(val) if not np.isnan(val) else np.nan

            unified_rows.append(row)

        df_unified = pd.DataFrame(unified_rows)
        csv_path_unified = os.path.join(
            paper_assets_dir, "unified_comparison_table.csv"
        )
        df_unified.to_csv(csv_path_unified, index=False)
        shutil.copy2(
            csv_path_unified,
            os.path.join(docs_paper_dir, "unified_comparison_table.csv"),
        )

        # 2. Compile LaTeX-ready Table
        default_thresholds = self._get_default_thresholds()
        if len(default_thresholds) >= 3:
            rep_thresholds = [default_thresholds[0], default_thresholds[len(default_thresholds)//2], default_thresholds[-1]]
        else:
            rep_thresholds = default_thresholds

        cols_headers = " & ".join([f"\\textbf{{Steps to $\\ge {t:.1f}$}}" for t in rep_thresholds])
        latex_table = f"""% LaTeX table generated by PPO Reward Shaping Analysis Pipeline
\\begin{{table*}}[t]
\\centering
\\caption{{Comparative Performance under Different Reward Shaping Strategies on {self.env_id}}}
\\label{{tab:reward_shaping_comparison}}
\\begin{{tabular}}{{l{"c" * (3 + len(rep_thresholds))}}}
\\hline
\\textbf{{Strategy}} & \\textbf{{Final Reward (Mean $\\pm$ SD)}} & \\textbf{{95\\% CI}} & {cols_headers} & \\textbf{{Train Time (s)}} \\\\
\\hline
"""
        import math

        for row in unified_rows:
            strat_name = row["Strategy"]
            final_rew = f"{row['Final Reward (Mean)']:.2f} $\\pm$ {row['Final Reward (Std)']:.2f}"
            ci = f"$\\pm$ {row['95% CI']:.2f}"

            steps_cols = []
            for t in rep_thresholds:
                val = row.get(f"Steps to R>={t}", np.nan)
                val_str = f"{val:.1f}" if isinstance(val, int | float) and not math.isnan(val) else "N/A"
                steps_cols.append(val_str)

            steps_str = " & ".join(steps_cols)
            train_time = f"{row['Mean Train Time (s)']:.1f}s"

            latex_table += f"{strat_name} & {final_rew} & {ci} & {steps_str} & {train_time} \\\\\n"

        latex_table += """\\hline
\\end{tabular}
\\end{table*}
"""
        tex_path = os.path.join(paper_assets_dir, "comparison_table.tex")
        with open(tex_path, "w") as f_tex:
            f_tex.write(latex_table)
        shutil.copy2(tex_path, os.path.join(docs_paper_dir, "comparison_table.tex"))

        # 3. Perform Pairwise Statistical Tests and Compile Unified Report
        import itertools

        report = f"""================================================================================
RL RESEARCH LAB: UNIFIED STATISTICAL EVALUATION MANUSCRIPT ASSETS
Strategies compared: {", ".join([s.upper() for s in strategies])}
Generated: {datetime.date.today().strftime("%Y-%m-%d")}
================================================================================

"""
        for strat1, strat2 in itertools.combinations(strategies, 2):
            results = self.perform_statistical_tests(strat1, strat2)

            json_path = os.path.join(
                paper_assets_dir, f"statistical_tests_{strat1}_vs_{strat2}.json"
            )
            with open(json_path, "w") as f_json:
                json.dump(results, f_json, indent=4)
            shutil.copy2(
                json_path,
                os.path.join(
                    docs_paper_dir, f"statistical_tests_{strat1}_vs_{strat2}.json"
                ),
            )

            fr = results["final_rewards"]
            tr = results["thresholds"]

            report += f"""--------------------------------------------------------------------------------
PAIRWISE COMPARISON: {strat1.upper()} vs. {strat2.upper()}
--------------------------------------------------------------------------------
- {strat1.upper()} raw rewards: {fr[f'{strat1}_values']}
- {strat2.upper()} raw rewards: {fr[f'{strat2}_values']}

Welch's Independent t-test:
  t-statistic = {fr['t_statistic']:.4f}
  p-value     = {fr['t_p_value']:.4e} (p < 0.05 is statistically significant)

Mann-Whitney U Rank Test:
  U-statistic = {fr['u_statistic']:.4f}
  p-value     = {fr['u_p_value']:.4e}

Effect Size:
  Cohen's d   = {fr['cohens_d']:.4f} (d > 0.8 represents large effect size)

Conclusion:
  {"The strategies differ significantly in final asymptotic performance." if fr['t_p_value'] < 0.05 else "No statistically significant difference in final performance was detected."}

Sample Efficiency (Steps to Thresholds):
| Threshold | {strat1.capitalize()} Mean (Steps) | {strat2.capitalize()} Mean (Steps) | Speedup Factor | t-p-value | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
            for t in self._get_default_thresholds():
                t_data = tr.get(t, {})
                m1 = t_data.get(f"{strat1}_mean", np.nan)
                m2 = t_data.get(f"{strat2}_mean", np.nan)
                speedup = (
                    m1 / m2
                    if m1 > 0 and m2 > 0 and not np.isnan(m1) and not np.isnan(m2)
                    else np.nan
                )
                p_val = t_data.get("t_p_value", 1.0)
                d_val = t_data.get("cohens_d", 0.0)

                m1_str = f"{m1:.1f}" if not np.isnan(m1) else "N/A"
                m2_str = f"{m2:.1f}" if not np.isnan(m2) else "N/A"
                speedup_str = f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A"

                report += f"| Reward {t:6.1f} | {m1_str:21s} | {m2_str:21s} | {speedup_str:14s} | {p_val:.4e} | {d_val:.4f} |\n"

            report += "\n"

        txt_path = os.path.join(paper_assets_dir, "statistical_summary.txt")
        with open(txt_path, "w") as f_txt:
            f_txt.write(report)
        shutil.copy2(txt_path, os.path.join(docs_paper_dir, "statistical_summary.txt"))
        print(f"Unified statistical manuscript assets generated in {paper_assets_dir}")
