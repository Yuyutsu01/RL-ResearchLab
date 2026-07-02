# Architecture Pipeline Diagram

The following Mermaid diagram visualizes the data flow, control relationships, and execution boundaries between Gymnasium, our wrappers, the SB3 PPO agent, the logger callbacks, and the automated documentation engine.

```mermaid
flowchart TD
    %% Define styles
    classDef config fill:#f9f,stroke:#333,stroke-width:2px;
    classDef env fill:#bbf,stroke:#333,stroke-width:2px;
    classDef agent fill:#f96,stroke:#333,stroke-width:2px;
    classDef log fill:#bfb,stroke:#333,stroke-width:2px;
    classDef doc fill:#fbb,stroke:#333,stroke-width:2px;

    subgraph ConfigPhase ["Configuration Phase"]
        YAML[configs/*.yaml]:::config -->|Parses| ConfigManager[utils/config.py]:::config
        ConfigManager -->|Provides Settings| Runner[experiments/runner.py]:::config
        SeedManager[utils/reproducibility.py]:::config -->|Seeds| Runner
    end

    subgraph EnvPhase ["Environment Construction"]
        Runner -->|Constructs| TrainEnv[Training Env Vec]:::env
        Runner -->|Constructs| EvalEnv[Evaluation Env Vec]:::env
        
        TrainEnv -->|Wrapped By| TrainWrapper[RewardShapingWrapper]:::env
        EvalEnv -->|Wrapped By| EvalWrapper[RewardShapingWrapper]:::env
        
        TrainWrapper -->|Applies| TargetShaper[Target RewardShaper]:::env
        EvalWrapper -->|Applies| IdentityShaper[IdentityRewardShaper]:::env
        
        TrainWrapper -->|Wrapped By| TrainMonitor[Monitor CSV Wrapper]:::env
        EvalWrapper -->|Wrapped By| EvalMonitor[Monitor CSV Wrapper]:::env
    end

    subgraph OptimizationPhase ["PPO Optimization Loop"]
        TrainMonitor -->|Obs & Shaped Reward| PPO[Stable-Baselines3 PPO]:::agent
        PPO -->|Action| TrainMonitor
        
        PPO -->|Step Callback| LogCallback[ResearchLoggingCallback]:::log
        TrainMonitor -->|Accumulates info| LogCallback
        LogCallback -->|Record Scalars| TB[TensorBoard Event Log]:::log
    end

    subgraph EvaluationPhase ["Periodic Evaluation Loop"]
        PPO -->|Deterministic Step| EvalCallback[EvalCallback]:::log
        EvalCallback -->|Evaluates| EvalMonitor
        EvalMonitor -->|Unbiased Raw Score| EvalCallback
        EvalCallback -->|Save Best Model| ModelCheckpoints[models/ checkpoints]:::agent
        EvalCallback -->|Save History| EvalNPZ[results/ evaluations.npz]:::log
    end

    subgraph AutoDocPhase ["Automated Research Notebook Documentation"]
        Runner -->|Saves metadata.json| Metadata[results/ metadata.json]:::log
        TrainMonitor -->|Writes monitor.csv| MonitorCSV[results/ monitor.csv]:::log
        
        Main[main.py CLI] -->|Triggers| AutoDoc[utils/autodoc.py Engine]:::doc
        
        Metadata --> AutoDoc
        MonitorCSV --> AutoDoc
        EvalNPZ --> AutoDoc
        TB --> AutoDoc
        
        AutoDoc -->|Aggregates & Writes| ExpFolder[docs/experiments/]:::doc
        AutoDoc -->|Appends Run| ExpIndex[docs/experiment_index.md]:::doc
        AutoDoc -->|Appends Log| Journal[docs/project_journal.md]:::doc
        AutoDoc -->|Copies Figures| Evidence[docs/evidence/]:::doc
    end
```
