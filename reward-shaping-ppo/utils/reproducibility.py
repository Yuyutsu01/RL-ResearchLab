import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Sets the random seed for Python, NumPy, PyTorch, and the environment.

    Args:
        seed: The integer seed to set.
        deterministic: If True, configures PyTorch backends to prioritize
            deterministic execution over speed.
        benchmark: If True, allows CuDNN to profile and select the fastest kernel
            algorithms (used when deterministic is False).
    """
    # Set seed for Python built-in random module
    random.seed(seed)

    # Set seed for NumPy operations
    np.random.seed(seed)

    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed)

    # Set seed for PyTorch GPU operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure deterministic algorithms in PyTorch
    if deterministic:
        # Forces PyTorch to use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Disable benchmarking (benchmarking chooses different algorithms dynamically)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic behavior in other libraries
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # Revert deterministic settings to defaults
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
