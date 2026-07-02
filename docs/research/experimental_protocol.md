# Experimental Protocol

This document establishes the official experimental protocol for executing and verifying reward shaping strategies within the PPO framework.

---

## 1. Constants and Control Variables

To maintain scientific control, the following variables must remain constant across all experiments:

| Variable | Control Setting |
| :--- | :--- |
| **RL Algorithm** | Proximal Policy Optimization (PPO) |
| **Policy Network** | Multi-Layer Perceptron (`MlpPolicy`) |
| **Network Architecture** | Actor: `[64, 64]`, Critic: `[64, 64]` |
| **Learning Rate** | `0.0003` (Constant) |
| **Batch Size** | `64` |
| **Steps per Update (`n_steps`)** | `2048` |
| **Number of Epochs (`n_epochs`)** | `10` |
| **Discount Factor ($\gamma$)** | `0.99` |
| **GAE Parameter ($\lambda$)** | `0.95` |
| **Entropy Coefficient** | `0.0` |
| **Environment** | `CartPole-v1` (Action space seeded) |
| **Random Seeds** | `[42, 43, 44, 45, 46]` (5 seeds) |
| **Total Timesteps** | `100,000` steps per seed |
| **Training Device** | `cpu` |

---

## 2. Execution Protocol

For each reward shaping strategy, the researcher (or automated pipeline) must perform the following steps:

1. **Write the Shaper**: Implement the strategy in `reward_functions/` and register it in the shaper factory.
2. **Create the YAML Config**: Define parameters in a YAML config under `configs/` matching the baseline structure.
3. **Execute the Experiment**: Run training across all 5 seeds via the CLI:
   ```bash
   python main.py --config configs/<config_name>.yaml --mode all
   ```
4. **Export Logs & Checkpoints**: Ensure the results directory creates folders for each seed containing:
   - `monitor.csv` (CSV log containing episode metrics)
   - `eval_monitor.csv` (evaluation log)
   - `evaluations.npz` (evaluation reward history)
   - `metadata.json` (runtime diagnostics)
   - `final_model.zip` & `best_model.zip` (checkpoints)
5. **Generate Visualizations**: Verify the `plots/` folder contains generated PNG and PDF curves.
6. **Compile Statistics**: Run the analyzer engine to update summary metrics.
