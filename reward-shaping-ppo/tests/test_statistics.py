import os
import json
import numpy as np
import pytest

from analysis.statistics import ExperimentAnalyzer


class TestFindSeedsForStrategy:
    def test_finds_existing_seeds(self, synthetic_results_tree):
        base_dir, env_id, strategies, seeds = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        for strategy in strategies:
            found = analyzer._find_seeds_for_strategy(strategy)
            assert len(found) == len(seeds)

    def test_returns_empty_for_nonexistent_strategy(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))
        found = analyzer._find_seeds_for_strategy("nonexistent_strategy")
        assert found == []

    def test_returns_empty_for_nonexistent_env(self, tmp_path):
        analyzer = ExperimentAnalyzer(env_id="NoSuchEnv-v99", base_dir=str(tmp_path))
        found = analyzer._find_seeds_for_strategy("identity")
        assert found == []


class TestLoadStrategyData:
    def test_returns_dict_with_expected_keys(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        data = analyzer.load_strategy_data("identity", grid_points=50)
        assert data is not None
        assert "steps" in data
        assert "original_reward" in data
        assert "shaped_reward" in data
        assert "episode_length" in data
        assert "raw_seeds_count" in data

    def test_grid_points_match(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        grid_points = 75
        data = analyzer.load_strategy_data("identity", grid_points=grid_points)
        assert len(data["steps"]) == grid_points

    def test_stats_have_correct_shape(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        data = analyzer.load_strategy_data("identity", grid_points=50)
        for metric in ["original_reward", "shaped_reward", "episode_length"]:
            stats = data[metric]
            for key in ["mean", "std", "median", "sem", "ci95", "min", "max"]:
                assert key in stats
                assert len(stats[key]) == 50

    def test_returns_none_for_missing_strategy(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))
        assert analyzer.load_strategy_data("nonexistent") is None

    def test_no_nan_in_output(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        data = analyzer.load_strategy_data("identity", grid_points=50)
        for metric in ["original_reward", "shaped_reward", "episode_length"]:
            for key in ["mean", "std", "median", "min", "max"]:
                assert not np.any(np.isnan(data[metric][key]))


class TestComputeSummaryStatistics:
    def test_returns_expected_keys(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        summary = analyzer.compute_summary_statistics("identity")
        assert summary is not None
        assert "strategy" in summary
        assert "num_seeds" in summary
        assert "final_unshaped_reward_mean" in summary
        assert "final_unshaped_reward_std" in summary
        assert "final_unshaped_reward_ci95" in summary
        assert "mean_training_time_seconds" in summary

    def test_correct_seed_count(self, synthetic_results_tree):
        base_dir, env_id, _, seeds = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        summary = analyzer.compute_summary_statistics("identity")
        assert summary["num_seeds"] == len(seeds)

    def test_saves_summary_json(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        analyzer.compute_summary_statistics("identity")
        summary_path = base_dir / "results" / env_id / "identity" / "summary.json"
        assert summary_path.exists()

        with open(summary_path) as f:
            saved = json.load(f)
        assert saved["strategy"] == "identity"

    def test_returns_none_for_missing_strategy(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))
        assert analyzer.compute_summary_statistics("nonexistent") is None

    def test_single_seed_handles_ci_gracefully(self, synthetic_single_strategy_tree):
        base_dir, env_id, strategy = synthetic_single_strategy_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        summary = analyzer.compute_summary_statistics(strategy)
        assert summary is not None
        assert summary["num_seeds"] == 1
        assert summary["final_unshaped_reward_ci95"] == 0.0


class TestGetTimestepsToThresholds:
    def test_returns_all_thresholds(self, synthetic_single_strategy_tree):
        base_dir, env_id, strategy = synthetic_single_strategy_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        thresholds = [100, 200, 300, 400, 500]
        result = analyzer.get_timesteps_to_thresholds(strategy, thresholds=thresholds)

        for t in thresholds:
            assert t in result
            assert "mean" in result[t]
            assert "std" in result[t]
            assert "values" in result[t]

    def test_threshold_values_are_monotonic(self, synthetic_single_strategy_tree):
        base_dir, env_id, strategy = synthetic_single_strategy_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        result = analyzer.get_timesteps_to_thresholds(strategy, thresholds=[100, 200, 300, 400, 500])
        prev_mean = 0
        for t in [100, 200, 300, 400, 500]:
            current_mean = result[t]["mean"]
            if not np.isnan(current_mean):
                assert current_mean >= prev_mean
                prev_mean = current_mean

    def test_returns_empty_for_missing_strategy(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))
        assert analyzer.get_timesteps_to_thresholds("nonexistent") == {}


class TestGetFinalEvaluationRewards:
    def test_returns_correct_count(self, synthetic_results_tree):
        base_dir, env_id, _, seeds = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        rewards = analyzer.get_final_evaluation_rewards("identity")
        assert len(rewards) == len(seeds)

    def test_rewards_are_finite(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        rewards = analyzer.get_final_evaluation_rewards("identity")
        for r in rewards:
            assert np.isfinite(r)


class TestPerformStatisticalTests:
    def test_returns_all_expected_keys(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        results = analyzer.perform_statistical_tests("identity", "dense")

        assert "final_rewards" in results
        assert "thresholds" in results

        fr = results["final_rewards"]
        assert "t_statistic" in fr
        assert "t_p_value" in fr
        assert "u_statistic" in fr
        assert "u_p_value" in fr
        assert "cohens_d" in fr

    def test_p_values_in_valid_range(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        results = analyzer.perform_statistical_tests("identity", "dense")
        fr = results["final_rewards"]

        if not np.isnan(fr["t_p_value"]):
            assert 0.0 <= fr["t_p_value"] <= 1.0
        if not np.isnan(fr["u_p_value"]):
            assert 0.0 <= fr["u_p_value"] <= 1.0

    def test_cohens_d_is_finite(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        results = analyzer.perform_statistical_tests("identity", "dense")
        assert np.isfinite(results["final_rewards"]["cohens_d"])


class TestGenerateStatisticalReport:
    def test_generates_report_files(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        analyzer.generate_statistical_report("identity", "dense")

        paper_dir = base_dir / "paper_assets"
        assert (paper_dir / "statistical_summary.txt").exists()
        assert (paper_dir / "comparison_table.csv").exists()
        assert (paper_dir / "statistical_tests_identity_vs_dense.json").exists()

    def test_report_contains_expected_sections(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        analyzer.generate_statistical_report("identity", "dense")

        txt_path = base_dir / "paper_assets" / "statistical_summary.txt"
        content = txt_path.read_text()

        assert "IDENTITY" in content
        assert "DENSE" in content
        assert "Welch's Independent t-test" in content
        assert "Mann-Whitney U Rank Test" in content
        assert "Cohen's d" in content


class TestGenerateComparisonReport:
    def test_returns_dataframe_with_rows(self, synthetic_results_tree):
        base_dir, env_id, strategies, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        df = analyzer.generate_comparison_report(strategies)
        assert len(df) == len(strategies)

    def test_saves_comparison_csv(self, synthetic_results_tree):
        base_dir, env_id, strategies, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        analyzer.generate_comparison_report(strategies)
        csv_path = base_dir / "results" / env_id / "comparison_report.csv"
        assert csv_path.exists()
