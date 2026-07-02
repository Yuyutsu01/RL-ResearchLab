# Quantitative Research Objectives

To answer our core research questions, this project aims to accomplish the following measurable scientific objectives:

---

## 1. Benchmarking Baseline Performance
* Establish a robust baseline by training standard PPO with an **Identity Reward** (no shaping) on `CartPole-v1` across **5 independent random seeds** for **100,000 steps** each.
* Record the baseline metrics:
  * Mean asymptotic reward at convergence.
  * Number of steps required to reach the maximum reward threshold (score of 500).
  * Run-to-run standard deviation and standard error of the mean (SEM).

## 2. Evaluation of Decoupled Shaping Strategies
* Evaluate and compare at least five distinct reward shaping categories:
  1. **Dense Reward Shaping**: Providing a continuous tracking signal at every time step (e.g., matching target angles or velocities).
  2. **Distance-Based Reward**: Adding penalties proportional to the distance from target or boundary states.
  3. **Potential-Based Reward Shaping (PBRS)**: Using difference potentials ($\gamma \Phi(s') - \Phi(s)$) to guarantee policy preservation.
  4. **Penalty-Based Reward**: Penalizing undesirable regions of the state space (e.g., high velocities or unstable angles) to enforce safety constraints.
  5. **Adaptive Reward**: Automatically adapting shaping scaling factors or potential offsets based on training progress (e.g., standard deviation of recent rewards).

## 3. Analysis of Policy Subversion and Optimality
* Verify whether each shaping strategy preserves the optimal policy of the original task.
* Uncover instances of policy subversion by demonstrating when an agent achieves a high *shaped* reward during training but performs poorly on the *unshaped evaluation objective*.
* Measure the divergence between training performance (shaped) and evaluation performance (original).

## 4. Quantification of Learning Dynamics
* Compare convergence speed-up factors:
  $$\text{Speed-up Factor} = \frac{\text{Baseline Steps to Threshold}}{\text{Shaped Steps to Threshold}}$$
* Analyze optimization stability by extracting policy gradient loss, value function loss, and policy entropy directly from TensorBoard logs. Check if shaping decreases policy entropy too quickly (leading to premature convergence).

## 5. Implementation of a Reproducible Research Archive
* Package the results into a standardized documentation catalog (`docs/experiments/` and `docs/evidence/`).
* Ensure all experiments are deterministic and reproducible on standard CPU/GPU hardware using fixed seeds.
