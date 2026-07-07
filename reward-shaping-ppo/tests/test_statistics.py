import json

import numpy as np

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

        result = analyzer.get_timesteps_to_thresholds(
            strategy, thresholds=[100, 200, 300, 400, 500]
        )
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


class TestNewStatisticalMethods:
    def test_calculate_auc(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))
        raw_aucs, norm_aucs = analyzer.calculate_auc("identity", max_steps=5000.0)
        assert len(raw_aucs) == 2
        assert len(norm_aucs) == 2
        for r, n in zip(raw_aucs, norm_aucs):
            assert r >= 0.0
            assert n >= 0.0
            assert np.isclose(n, r / 5000.0)

    def test_calculate_cliffs_delta(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        delta = analyzer.calculate_cliffs_delta(g1, g2)
        assert delta == 0.0

        g3 = [6.0, 7.0, 8.0, 9.0, 10.0]
        delta_pos = analyzer.calculate_cliffs_delta(g3, g1)
        assert delta_pos == 1.0

        delta_neg = analyzer.calculate_cliffs_delta(g1, g3)
        assert delta_neg == -1.0

    def test_calculate_bootstrap_ci(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        data = [10.0, 10.0, 10.0]
        lower, upper = analyzer.calculate_bootstrap_ci(data, num_resamples=100)
        assert lower == 10.0
        assert upper == 10.0

        data_var = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower_v, upper_v = analyzer.calculate_bootstrap_ci(data_var, num_resamples=100)
        assert lower_v <= upper_v
        assert 1.0 <= lower_v <= 5.0
        assert 1.0 <= upper_v <= 5.0

    def test_apply_benjamini_hochberg(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        p_values = [0.001, 0.01, 0.04, 0.5, 0.8]
        reject = analyzer.apply_benjamini_hochberg(p_values, alpha=0.05)
        assert len(reject) == 5
        assert reject[0] is True
        assert reject[1] is True
        assert reject[2] is False
        assert reject[3] is False
        assert reject[4] is False

    def test_get_timesteps_to_fractional_thresholds(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(base_dir))

        results = analyzer.get_timesteps_to_fractional_thresholds("identity")
        for f in [0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 1.00]:
            assert f in results
            assert "mean" in results[f]
            assert "std" in results[f]
            assert "values" in results[f]
