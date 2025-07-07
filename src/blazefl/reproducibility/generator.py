import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def setup_reproducibility(seed: int) -> None:
    """
    Configures the environment-level settings for deterministic behavior.

    This function sets the `PYTHONHASHSEED` for consistent hash-based operations
    and configures PyTorch's cuDNN backend to use deterministic algorithms.
    Call this at the start of your script for a stable environment.

    Args:
        seed: The seed value to use for the hash seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class RNGSuite:
    """
    A container for a suite of isolated random number generators.

    This class holds independent, seeded generator objects for each library.
    Using an `RNGSuite` instance allows for randomness control that is self-contained
    and does not interfere with the global state, making it suitable for components or
    libraries that need their own random stream.

    Attributes:
        python: An isolated `random.Random` generator.
        numpy: An isolated `numpy.random.Generator` instance.
        torch_cpu: A `torch.Generator` for CPU operations.
        torch_cuda: A `torch.Generator` for CUDA operations, if available.
    """

    python: random.Random
    numpy: np.random.Generator
    torch_cpu: torch.Generator
    torch_cuda: torch.Generator | None = None


def create_rng_suite(seed: int) -> RNGSuite:
    """
    Creates a new suite of isolated random number generators from a single seed.

    This is a convenience factory function to instantiate `RNGSuite` with all its
    generators properly seeded and ready for use.

    Args:
        seed: The master seed to initialize all generators in the suite.

    Returns:
        A new `RNGSuite` instance.
    """
    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    torch_cpu_rng = torch.Generator(device="cpu").manual_seed(seed)

    torch_cuda_rng = None
    if torch.cuda.is_available():
        torch_cuda_rng = torch.Generator("cuda").manual_seed(seed)

    return RNGSuite(
        python=python_rng,
        numpy=numpy_rng,
        torch_cpu=torch_cpu_rng,
        torch_cuda=torch_cuda_rng,
    )
