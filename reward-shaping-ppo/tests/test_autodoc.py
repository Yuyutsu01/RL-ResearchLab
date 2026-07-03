import os
import json
import pytest

from utils.autodoc import AutoDocManager


class TestAutoDocManagerStructure:
    def test_creates_docs_structure(self, synthetic_results_tree):
        base_dir, env_id, strategies, seeds = synthetic_results_tree
        package_dir = base_dir / "reward-shaping-ppo"
        package_dir.mkdir(exist_ok=True)

        import shutil
        for d in ["results", "plots", "paper_assets", "configs"]:
            src = base_dir / d
            dst = package_dir / d
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))

        manager = AutoDocManager(base_dir=str(package_dir))
        manager._create_docs_structure()

        docs_dir = os.path.join(manager.workspace_root, "docs")
        expected_dirs = [
            "research", "architecture", "decisions", "experiments",
            "results", "evidence", "literature", "meeting_notes", "paper"
        ]
        for subdir in expected_dirs:
            assert os.path.isdir(os.path.join(docs_dir, subdir))

    def test_document_experiment_creates_files(self, synthetic_results_tree):
        base_dir, env_id, strategies, seeds = synthetic_results_tree
        package_dir = base_dir / "reward-shaping-ppo"
        package_dir.mkdir(exist_ok=True)

        import shutil
        for d in ["results", "plots", "paper_assets", "configs"]:
            src = base_dir / d
            dst = package_dir / d
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))

        manager = AutoDocManager(base_dir=str(package_dir))

        from analysis.statistics import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(package_dir))
        analyzer.compute_summary_statistics("identity")

        manager.document_experiment(env_id=env_id, strategy="identity")

        docs_dir = os.path.join(manager.workspace_root, "docs")
        assert os.path.exists(os.path.join(docs_dir, "experiments", "identity", "overview.md"))
        assert os.path.exists(os.path.join(docs_dir, "experiments", "identity", "metrics.md"))
        assert os.path.exists(os.path.join(docs_dir, "results", "identity", "summary.md"))
        assert os.path.exists(os.path.join(docs_dir, "experiment_index.md"))
        assert os.path.exists(os.path.join(docs_dir, "project_journal.md"))

    def test_document_experiment_copies_raw_data(self, synthetic_results_tree):
        base_dir, env_id, strategies, seeds = synthetic_results_tree
        package_dir = base_dir / "reward-shaping-ppo"
        package_dir.mkdir(exist_ok=True)

        import shutil
        for d in ["results", "plots", "paper_assets", "configs"]:
            src = base_dir / d
            dst = package_dir / d
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))

        manager = AutoDocManager(base_dir=str(package_dir))

        from analysis.statistics import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(package_dir))
        analyzer.compute_summary_statistics("identity")

        manager.document_experiment(env_id=env_id, strategy="identity")

        docs_dir = os.path.join(manager.workspace_root, "docs")
        raw_dir = os.path.join(docs_dir, "experiments", "identity", "raw")

        seed_dirs = [d for d in os.listdir(raw_dir) if d.startswith("seed_")]
        assert len(seed_dirs) > 0

        first_seed = os.path.join(raw_dir, seed_dirs[0])
        assert os.path.exists(os.path.join(first_seed, "monitor.csv"))
        assert os.path.exists(os.path.join(first_seed, "metadata.json"))

    def test_experiment_index_updated(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        package_dir = base_dir / "reward-shaping-ppo"
        package_dir.mkdir(exist_ok=True)

        import shutil
        for d in ["results", "plots", "paper_assets", "configs"]:
            src = base_dir / d
            dst = package_dir / d
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))

        manager = AutoDocManager(base_dir=str(package_dir))

        from analysis.statistics import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(package_dir))
        analyzer.compute_summary_statistics("identity")

        manager.document_experiment(env_id=env_id, strategy="identity")

        docs_dir = os.path.join(manager.workspace_root, "docs")
        index_path = os.path.join(docs_dir, "experiment_index.md")

        with open(index_path) as f:
            content = f.read()

        assert "EXP-TESTENV-V0-IDENTITY" in content

    def test_does_not_crash_on_unknown_strategy(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        package_dir = base_dir / "reward-shaping-ppo"
        package_dir.mkdir(exist_ok=True)

        import shutil
        for d in ["results", "plots", "paper_assets", "configs"]:
            src = base_dir / d
            dst = package_dir / d
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))

        custom_dir = package_dir / "results" / env_id / "custom" / "seed_42"
        custom_dir.mkdir(parents=True, exist_ok=True)

        manager = AutoDocManager(base_dir=str(package_dir))
        manager.document_experiment(env_id=env_id, strategy="custom")

        docs_dir = os.path.join(manager.workspace_root, "docs")
        overview = os.path.join(docs_dir, "experiments", "custom", "overview.md")
        assert os.path.exists(overview)

        with open(overview) as f:
            content = f.read()
        assert "N/A" in content or "Custom" in content


class TestAutoDocManagerIdempotency:
    def test_double_run_no_duplicate_journal_entries(self, synthetic_results_tree):
        base_dir, env_id, _, _ = synthetic_results_tree
        package_dir = base_dir / "reward-shaping-ppo"
        package_dir.mkdir(exist_ok=True)

        import shutil
        for d in ["results", "plots", "paper_assets", "configs"]:
            src = base_dir / d
            dst = package_dir / d
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))

        manager = AutoDocManager(base_dir=str(package_dir))

        from analysis.statistics import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer(env_id=env_id, base_dir=str(package_dir))
        analyzer.compute_summary_statistics("identity")

        manager.document_experiment(env_id=env_id, strategy="identity")
        manager.document_experiment(env_id=env_id, strategy="identity")

        docs_dir = os.path.join(manager.workspace_root, "docs")
        index_path = os.path.join(docs_dir, "experiment_index.md")

        with open(index_path) as f:
            content = f.read()

        count = content.count("EXP-TESTENV-V0-IDENTITY")
        assert count == 1
