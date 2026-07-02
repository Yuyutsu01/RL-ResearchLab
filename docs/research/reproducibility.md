# Reproducibility Standards

A primary goal of this research project is ensuring that every experiment is fully reproducible by independent researchers on different machines. This document outlines the protocols we enforce to guarantee reproducibility.

---

## 1. Multi-Level Random Seeding

We manage randomness across all software layers by initializing the following libraries with identical seeds:

1. **Python Built-in `random`**: Seeds the Python standard library's random number generator.
2. **NumPy `np.random`**: Seeds all NumPy array initializations, matrix operations, and noise processes.
3. **PyTorch `torch.manual_seed`**: Seeds the neural network weights initialization and gradient optimization updates (backpropagation stochasticity).
4. **Gymnasium Spaces**: Seeds the environment's action and observation spaces to ensure the starting state distributions and action samplings are consistent across runs.
5. **Stable-Baselines3**: Seeds the internal exploration noise and action selection policies.

---

## 2. Deterministic Deep Learning Backends

By default, PyTorch uses optimized CUDA and CuDNN backends that execute operations in parallel. Because floating-point operations are non-associative, minor variations in computation orders can lead to different round-off errors, causing weights to diverge slightly over thousands of updates.

To prevent this divergence:
- When `reproducibility.deterministic: true` is configured:
  - We force PyTorch to use deterministic algorithms where possible: `torch.use_deterministic_algorithms(True, warn_only=True)`.
  - We disable CuDNN auto-tuner benchmarking: `torch.backends.cudnn.benchmark = False`.
  - We force CuDNN to prioritize deterministic algorithms: `torch.backends.cudnn.deterministic = True`.
  - We configure CUBLAS workspace limits: `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`.

---

## 3. Auditing Environment Configurations

To ensure there are no discrepancies in dependencies, each experiment run automatically copies:
- The exact **YAML hyperparameter configuration** into the seed results folder.
- A **`metadata.json`** diagnostics log containing:
  - The runtime training time.
  - The device type used (e.g., `"cpu"` or `"cuda"`).
  - The deterministic execution state.
  - The exact final model weights filepath.
