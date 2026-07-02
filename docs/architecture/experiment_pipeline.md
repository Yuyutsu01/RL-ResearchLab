# Experiment Execution Pipeline

This document explains the technical execution lifecycle of an experiment, from the initial CLI invocation to the generation of visual assets and final folder archiving.

---

## The Execution Lifecycle

When the user runs the primary entry point:
```bash
python main.py --config configs/cartpole_baseline.yaml --mode all
```

The system executes the following steps:

```
  1. main.py parses arguments and loads YAML via Config Manager (utils/config.py)
                              |
                              v
  2. Setup output directories: results/, models/, logs/, plots/
                              |
                              v
  3. Loop through configured seeds (e.g. 42, 43, 44, 45, 46):
     a. Apply random seeds (utils/reproducibility.py)
     b. Instantiate shaped training environment & raw evaluation environment
     c. Instantiate SB3 PPO model with hyperparameters and device settings
     d. Attach callbacks: ResearchLoggingCallback & EvalCallback
     e. Train model for total timesteps, saving best models and evaluations.npz
     f. Save final model checkpoint and results/metadata.json
                              |
                              v
  4. Run Statistical Analysis (analysis/statistics.py):
     a. Read monitor.csv files across all seeds
     b. Interpolate and align rewards onto common steps grid
     c. Compute rolling averages, means, standard deviations, and 95% CIs
     v
  5. Generate Visualizations (utils/plotting.py):
     a. Draw learning curves, evaluation curves, and single-seed loss charts
     b. Save plots as PNG (web) and PDF (vector graphics) under plots/
                              |
                              v
  6. Run AutoDoc Engine (utils/autodoc.py):
     a. Copy all raw run logs, configs, and charts to docs/experiments/[strategy]/
     b. Generate metrics.md and overview.md summary files
     c. Update docs/experiment_index.md and docs/project_journal.md logs
```
---

## Key Verification Mechanisms

* **Unbiased Evaluation Checkpoints**: The PPO policy is periodically frozen and evaluated on the raw unshaped evaluation environment. This guarantees that model checkpoint selection is based on the true objective, preventing policies that exploit shaped proxy goals from being marked as the "best" model.
* **Metadata Audit Trails**: Each completed run writes a `metadata.json` containing runtime device configurations and exact seeds, serving as a scientific audit trail for the experiment.
