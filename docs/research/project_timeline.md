# Project Research Timeline

This document tracks the milestones of our comparative study from inception to final paper publication.

---

## Phase 1: Core Framework & Baseline (Days 1–8)
- [x] **Day 1**: Initialized the research repository and established pinned library dependencies.
- [x] **Day 3**: Designed the modular `RewardShaper` interface and `IdentityRewardShaper` pass-through class.
- [x] **Day 5**: Developed the custom `RewardShapingWrapper` and integrated it with standard Gymnasium classical control environments.
- [x] **Day 6**: Implemented `ResearchLoggingCallback` for streaming unshaped vs. shaped metrics to TensorBoard.
- [x] **Day 7**: Configured the multi-seed `ExperimentRunner` and verified determinism.
- [x] **Day 8**: Completed the 5-seed baseline PPO experiment on `CartPole-v1`. Generated initial statistical reports and loss charts.

## Phase 2: Documentation System & Research Lab (Days 9–12)
- [x] **Day 9**: Established the `docs/` structure, outlining hypotheses, protocols, and methodology.
- [x] **Day 10**: Formulated the Architecture Decision Records (ADRs 001–006) justifying core design trade-offs.
- [x] **Day 11**: Implemented the automated documentation engine (`AutoDocManager`) to package results.
- [x] **Day 12**: Integrated the documentation engine with the main CLI, successfully archiving the baseline experiment.

## Phase 3: Shaped Strategy Implementations (Days 13–24)
- [ ] **Day 13**: Implement and evaluate **Dense Reward Shaping** on `CartPole-v1`.
- [ ] **Day 15**: Implement and evaluate **Distance-Based Reward Shaping**.
- [ ] **Day 18**: Implement and evaluate **Potential-Based Reward Shaping (PBRS)**.
- [ ] **Day 20**: Implement and evaluate **Penalty-Based Reward Shaping**.
- [ ] **Day 22**: Implement and evaluate **Adaptive Reward Shaping**.
- [ ] **Day 24**: Generate comparative cross-strategy benchmarking report and final paper plots.

## Phase 4: Scaling & Paper Draft (Days 25–30)
- [ ] **Day 25**: Scale testing to more complex environments (`MountainCar-v0`, `LunarLander-v2`).
- [ ] **Day 28**: Compile literature reviews and structure LaTeX paper assets under `docs/paper/`.
- [ ] **Day 30**: Finalize comparative manuscript draft for publication submission.
