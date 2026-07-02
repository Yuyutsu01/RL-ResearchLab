# PPO Reward Shaping Research Notebook & Documentation

Welcome to the central documentation archive and research journal for the **PPO Reward Shaping Strategies** comparative experimental study.

This folder serves as a comprehensive, self-contained record of the project's scientific inquiry, architectural decisions, and empirical results. Any researcher or developer should be able to review these files to understand why this project exists, the methodology, the experiments conducted, and the conclusions reached, without needing to dig into the source code.

---

## Navigation Directory

To explore the research lab, follow the links below:

### 1. Research Notebook (`docs/research/`)
This directory outlines the scientific foundations of the study:
* [Core Research Question](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/research/research_question.md): The primary problem we are investigating.
* [Quantitative Objectives](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/research/objectives.md): Specific metrics and project targets.
* [Theoretical Hypotheses](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/research/hypotheses.md): Predictions for each reward shaping strategy.
* [Methodology & Architecture Decoupling](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/research/methodology.md): The design of the experimental apparatus.
* [Experimental Protocol](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/research/experimental_protocol.md): Hyperparameters, baseline controls, and evaluation intervals.
* [Reproducibility Standards](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/research/reproducibility.md): Seed settings, environment configurations, and GPU determinism limits.
* [Project Timeline](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/research/project_timeline.md): Development milestones and phases.
* [Future Work](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/research/future_work.md): Open research avenues and proposed follow-up studies.

### 2. Architecture & Component Reference (`docs/architecture/`)
Technical blueprints detailing how the framework is built:
* [System Overview](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/architecture/system_overview.md): Decoupled component descriptions and dependencies.
* [Architecture Diagram](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/architecture/architecture_diagram.md): Diagram of the training-evaluation-shaping pipeline.
* [Reward Interception Pipeline](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/architecture/reward_pipeline.md): How `RewardShapingWrapper` modifies rewards dynamically.
* [Experiment Execution Flow](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/architecture/experiment_pipeline.md): Step-by-step experiment orchestration.
* [Component Reference](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/architecture/component_reference.md): API document mapping classes, functions, and interfaces.

### 3. Architecture Decision Records (`docs/decisions/` / ADRs)
Technical design justifications, explaining the trade-offs of chosen dependencies and constraints:
* [ADR-001: Stable-Baselines3 Selection](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/decisions/ADR-001-Why-Stable-Baselines3.md)
* [ADR-002: Gymnasium Migration Selection](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/decisions/ADR-002-Why-Gymnasium.md)
* [ADR-003: Core Identity Baseline Choice](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/decisions/ADR-003-Why-Identity-Baseline.md)
* [ADR-004: Standardized Multi-Random Seeds](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/decisions/ADR-004-Why-Deterministic-Seeds.md)
* [ADR-005: Unbiased Separate Evaluation Environments](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/decisions/ADR-005-Why-Separate-Evaluation-Environment.md)
* [ADR-006: Modular Decoupled Wrapper Interface](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/decisions/ADR-006-Reward-Shaping-Design.md)

### 4. Running Ledger
Active indices tracking development status:
* [Project Journal](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/project_journal.md): A chronological development log detailing the history of modifications.
* [Experiment Index](file:///c:/Users/shiva/OneDrive/Desktop/projects/RL-ResearchLab/docs/experiment_index.md): Markdown index cataloging all completed experiments, configurations, and result files.

### 5. Experiment Runs (`docs/experiments/`)
Auto-generated and curated details of each evaluated reward shaping strategy:
* `baseline_identity/`: The baseline configuration.
* `dense_reward/`: *(Future)* Dense reward shaping.
* `pbrs/`: *(Future)* Potential-Based Reward Shaping.
* `distance_reward/`: *(Future)* Position/Distance-based reward functions.
* `penalty_reward/`: *(Future)* Constraint-based penalty rewards.
* `adaptive_reward/`: *(Future)* Time-varying or state-varying adaptive rewards.

Each folder includes an overview, a metrics table, generated curve sheets, and raw data outputs.
