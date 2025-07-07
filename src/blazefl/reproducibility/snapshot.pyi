import numpy as np
import numpy.typing as npt
import torch
from blazefl.reproducibility.generator import setup_reproducibility as setup_reproducibility
from dataclasses import dataclass
from typing import Any

def seed_everything(seed: int) -> None: ...

@dataclass
class RandomStateSnapshot:
    environ: str
    python: tuple[Any, ...]
    numpy: tuple[str, npt.NDArray[np.uint32], int, int, float]
    torch_cpu: torch.Tensor
    torch_cuda: torch.Tensor | None
    @classmethod
    def capture(cls) -> RandomStateSnapshot: ...
    @staticmethod
    def restore(snapshot: RandomStateSnapshot) -> None: ...
