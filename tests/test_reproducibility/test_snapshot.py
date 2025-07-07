import random

import numpy as np
import pytest
import torch

from src.blazefl.reproducibility import RandomStateSnapshot, seed_everything


def test_seed_everything() -> None:
    seed = 42
    seed_everything(seed)

    py_rand_val = random.random()
    np_rand_val = np.random.rand()
    torch_rand_val = torch.rand(1)

    seed_everything(seed)
    assert py_rand_val == random.random()
    assert np.allclose(np_rand_val, np.random.rand())
    assert torch.allclose(torch_rand_val, torch.rand(1))


def test_random_state_snapshot_cpu() -> None:
    device = "cpu"
    seed = 123
    seed_everything(seed)

    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1, device=device)

    snapshot = RandomStateSnapshot.capture()

    py_rand_val_before = random.random()
    np_rand_val_before = np.random.rand()
    torch_rand_val_before = torch.rand(1, device=device)

    RandomStateSnapshot.restore(snapshot)

    assert py_rand_val_before == random.random()
    assert np.allclose(np_rand_val_before, np.random.rand())
    assert torch.allclose(torch_rand_val_before, torch.rand(1, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_random_state_snapshot_cuda() -> None:
    device = "cuda"
    seed = 123
    seed_everything(seed)

    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1, device=device)

    snapshot = RandomStateSnapshot.capture()

    py_rand_val_before = random.random()
    np_rand_val_before = np.random.rand()
    torch_rand_val_before = torch.rand(1, device=device)

    RandomStateSnapshot.restore(snapshot)

    assert py_rand_val_before == random.random()
    assert np.allclose(np_rand_val_before, np.random.rand())
    assert torch.allclose(torch_rand_val_before, torch.rand(1, device=device))
