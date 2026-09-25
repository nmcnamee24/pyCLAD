"""Keep video reproducibility tests from changing other modalities' RNG state."""

import os
import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def restore_random_state():
    """Restore the process-wide state modified by the public CLI seed helper."""
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if cublas_config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
