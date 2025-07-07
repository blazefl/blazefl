import numpy as np
import random
import torch
from dataclasses import dataclass

def setup_reproducibility(seed: int) -> None: ...

@dataclass
class RNGSuite:
    python: random.Random
    numpy: np.random.Generator
    torch_cpu: torch.Generator
    torch_cuda: torch.Generator | None = ...

def create_rng_suite(seed: int) -> RNGSuite: ...
