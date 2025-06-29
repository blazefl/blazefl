import numpy as np
import random
import torch
from dataclasses import dataclass

def seed_everything(seed: int, device: str) -> None: ...

@dataclass
class CUDARandomState:
    manual_seed: int
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_rng_state: torch.Tensor

@dataclass
class RandomState:
    @classmethod
    def get_random_state(cls, device: str) -> RandomState: ...
    @staticmethod
    def set_random_state(random_state: RandomState) -> None: ...

def seed_worker(worker_id: int): ...
def setup_reproducibility(seed: int) -> None: ...

@dataclass
class RNGSuite:
    python: random.Random
    numpy: np.random.Generator
    torch_cpu: torch.Generator
    torch_cuda: torch.Generator | None = ...

def create_rng_suite(seed: int) -> RNGSuite: ...
