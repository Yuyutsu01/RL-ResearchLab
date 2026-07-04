import argparse
import os

# ci-trigger
from analysis.statistics import ExperimentAnalyzer
from utils.plotting import (
    plot_learning_curves,
    plot_evaluation_curves,
    plot_training_losses,
)
from utils.config import Config
from utils.plotting import plot_evaluation_curves, plot_learning_curves, plot_training_losses



def main():
    parser = argparse.ArgumentParser(description="Modular Research Framework: Reward Shaping Strategies for PPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cartpole_baseline.yaml",
        help="Path to the YAML configuration file (default: configs/cartpole_baseline.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "analyze", "plot", "all"],
        default="all",
        help="Execution mode: train (run PPO training), analyze (run statistics), plot (render curves), all (run all phases)",
    )

    args = parser.parse_args()

    # 1. Resolve configuration paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(args.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    print(f"Loading config: {config_path}")
    config = Config.from_yaml(config_path)
    env_id = config.experiment.env_id
    strategy = config.reward_shaping.strategy
    seeds = config.experiment.seeds

    # 2. Phase: Training
    if args.mode in ["train", "all"]:
        print(
            f"\n>>> Running Phase: PPO Training on '{env_id}' using strategy '{strategy}'"
        )
        runner = ExperimentRunner(config_path=config_path, base_dir=current_dir)
        runner.run_all()

    # 3. Phase: Analysis
    if args.mode in ["analyze", "all"]:
        print(f"\n>>> Running Phase: Statistical Analysis for env '{env_id}'")
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=current_dir)

        # Dynamically discover all strategies that have trained results
        results_dir = os.path.join(current_dir, "results", env_id)
        strategies_to_analyze = []
        if os.path.exists(results_dir):
            for name in os.listdir(results_dir):
                if os.path.isdir(os.path.join(results_dir, name)) and name != "plots":
                    strategies_to_analyze.append(name)
        if not strategies_to_analyze:
            strategies_to_analyze = ["identity"]
            if strategy != "identity":
                strategies_to_analyze.append(strategy)

        print(f"Discovered strategies for analysis: {strategies_to_analyze}")

        print("\nComputing Summary Statistics...")
        for strat in strategies_to_analyze:
            try:
                summary = analyzer.compute_summary_statistics(strat)
                if summary:
                    print(f"\n--- Strategy: {strat.upper()} ---")
                    print(f"  Seeds evaluated: {summary['num_seeds']}")
                    print(
                        f"  Final Reward (Mean): {summary['final_unshaped_reward_mean']:.2f}"
                    )
                    print(
                        f"  Final Reward (Std):  {summary['final_unshaped_reward_std']:.2f}"
                    )
                    print(
                        f"  Final Reward (95% CI): ±{summary['final_unshaped_reward_ci95']:.2f}"
                    )
                    print(
                        f"  Mean Training Time:  {summary['mean_training_time_seconds']:.1f}s"
                    )
            except Exception:
                print(
                    f"Note: Strategy '{strat}' statistics not loaded (this is expected if it hasn't trained yet)."
                )

        # Generate comparative dataframe
        try:
            report_df = analyzer.generate_comparison_report(strategies_to_analyze)
            print("\nComparative Summary:")
            print(
                report_df.to_markdown(index=False)
                if hasattr(report_df, "to_markdown")
                else report_df
            )
        except Exception as e:
            print(f"Note: Comparative report not fully generated: {e}")

        # Run statistical tests and generate manuscript assets
        if len(strategies_to_analyze) > 1:
            print("\nGenerating unified comparative and LaTeX manuscript assets...")
            try:
                analyzer.generate_latex_and_manuscript_assets(strategies_to_analyze)
            except Exception as e:
                print(f"Failed to generate comparative manuscript assets: {e}")

    # 4. Phase: Visualization (Plotting)
    if args.mode in ["plot", "all"]:
        print(f"\n>>> Running Phase: Plotting Figures for env '{env_id}'")

        # Dynamically discover all strategies that have trained results for plotting
        results_dir = os.path.join(current_dir, "results", env_id)
        strategies_to_plot = []
        if os.path.exists(results_dir):
            for name in os.listdir(results_dir):
                if os.path.isdir(os.path.join(results_dir, name)) and name != "plots":
                    strategies_to_plot.append(name)
        if not strategies_to_plot:
            strategies_to_plot = ["identity"]
            if strategy != "identity":
                strategies_to_plot.append(strategy)

        print(f"Discovered strategies for plotting: {strategies_to_plot}")

        print("Rendering Learning Curves (Original, Shaped, Episode Length)...")
        plot_learning_curves(
            env_id=env_id,
            strategies=strategies_to_plot,
            base_dir=current_dir,
            grid_points=100,
            rolling_window=10,
        )

        print("Rendering Evaluation curves...")
        plot_evaluation_curves(
            env_id=env_id, strategies=strategies_to_plot, base_dir=current_dir
        )

        print("Rendering single-seed Loss metrics...")
        for seed in seeds:
            try:
                plot_training_losses(
                    env_id=env_id, strategy=strategy, seed=seed, base_dir=current_dir
                )
            except Exception as e:
                print(f"Skipped loss plotting for seed {seed} due to: {e}")

        print("\nVisualizations completed. Plots saved to 'plots/' directory.")

        # Document completed experiment in docs/ structure
        print("\n>>> Running Phase: Automated Research Lab Archiving")
        try:
            from utils.autodoc import AutoDocManager

            doc_manager = AutoDocManager(base_dir=current_dir)
            doc_manager.document_experiment(env_id=env_id, strategy=strategy)
            print("Research documentation updated successfully in 'docs/' directory.")
        except Exception as e:
            print(f"Failed to compile research documentation due to: {e}")


if __name__ == "__main__":
    main()
