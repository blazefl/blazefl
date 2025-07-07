import numpy as np
import numpy.typing as npt
import random
import torch
from dataclasses import dataclass
from typing import Any

def seed_everything(seed: int) -> None: ...

@dataclass
class RandomStateSnapshot:
    environ: str
    python: tuple[Any, ...]
    numpy: tuple[str, npt.NDArray[np.uint32], int, int, float]
    torch_cpu: torch.Tensor
    torch_cpu_seed: int
    torch_cuda: torch.Tensor | None
    torch_cuda_seed: int | None
    @classmethod
    def capture(cls) -> RandomStateSnapshot: ...
    @staticmethod
    def restore(snapshot: RandomStateSnapshot) -> None: ...

def setup_reproducibility(seed: int) -> None: ...

@dataclass
class RNGSuite:
    python: random.Random
    numpy: np.random.Generator
    torch_cpu: torch.Generator
    torch_cuda: torch.Generator | None = ...

def create_rng_suite(seed: int) -> RNGSuite: ...
