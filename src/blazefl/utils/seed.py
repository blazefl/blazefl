import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


def seed_everything(seed: int) -> None:
    """
    Seeds the global random number generators for all relevant libraries.

    This function sets a single seed for Python's `random` module, NumPy, and PyTorch
    to ensure that results are consistent across runs. It directly manipulates the
    global state of these libraries.

    Args:
        seed: The integer value for the seed.
    """
    setup_reproducibility(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class RandomStateSnapshot:
    """
    A snapshot of the global random state for major libraries.

    This class provides a mechanism to capture the exact state of the global random
    number generators and later restore it. This is useful for scenarios that require
    resuming a process from a specific point in time with the identical sequence of
    random numbers.

    Attributes:
        environ: The value of the `PYTHONHASHSEED` environment variable.
        python: The internal state of Python's `random` module.
        numpy: The internal state of NumPy's legacy random number generator.
        torch_cpu: The RNG state of the PyTorch CPU generator.
        torch_cpu_seed: The initial seed of the PyTorch CPU generator.
        torch_cuda: The RNG state of the PyTorch CUDA generator, if available.
        torch_cuda_seed: The initial seed of the PyTorch CUDA generator, if available.
    """

    environ: str
    python: tuple[Any, ...]
    numpy: tuple[str, npt.NDArray[np.uint32], int, int, float]
    torch_cpu: torch.Tensor
    torch_cpu_seed: int
    torch_cuda: torch.Tensor | None
    torch_cuda_seed: int | None

    @classmethod
    def capture(cls) -> "RandomStateSnapshot":
        """
        Captures the current global random state.

        Returns:
            A `RandomStateSnapshot` instance containing the captured states.
        """
        _environ = os.environ["PYTHONHASHSEED"]
        _python = random.getstate()
        _numpy = np.random.get_state(legacy=True)
        assert isinstance(_numpy, tuple)
        _torch_cpu = torch.get_rng_state()
        _torch_cpu_seed = torch.initial_seed()

        snapshot = cls(
            _environ, _python, _numpy, _torch_cpu, _torch_cpu_seed, None, None
        )
        if torch.cuda.is_available():
            snapshot.torch_cuda = torch.cuda.get_rng_state()
            snapshot.torch_cuda_seed = torch.cuda.initial_seed()
        return snapshot

    @staticmethod
    def restore(snapshot: "RandomStateSnapshot") -> None:
        """
        Restores the global random state from a snapshot object.

        Args:
            snapshot: The `RandomStateSnapshot` to restore from.
        """
        os.environ["PYTHONHASHSEED"] = snapshot.environ
        random.setstate(snapshot.python)
        np.random.set_state(snapshot.numpy)
        torch.set_rng_state(snapshot.torch_cpu)
        torch.manual_seed(snapshot.torch_cpu_seed)
        if snapshot.torch_cuda is not None and snapshot.torch_cuda_seed is not None:
            torch.cuda.set_rng_state(snapshot.torch_cuda)
            torch.cuda.manual_seed(snapshot.torch_cuda_seed)


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
