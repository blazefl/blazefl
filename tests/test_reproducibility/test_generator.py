import random

import numpy as np
import torch

from src.blazefl.reproducibility import create_rng_suite


def test_rng_suite_creation_and_reproducibility() -> None:
    seed = 42

    rng_suite1 = create_rng_suite(seed)
    py_val1 = rng_suite1.python.random()
    np_val1 = rng_suite1.numpy.random()
    torch_cpu_val1 = torch.rand(1, generator=rng_suite1.torch_cpu)
    torch_cuda_val1 = None
    if rng_suite1.torch_cuda:
        torch_cuda_val1 = torch.rand(1, generator=rng_suite1.torch_cuda)

    rng_suite2 = create_rng_suite(seed)
    py_val2 = rng_suite2.python.random()
    np_val2 = rng_suite2.numpy.random()
    torch_cpu_val2 = torch.rand(1, generator=rng_suite2.torch_cpu)
    torch_cuda_val2 = None
    if rng_suite2.torch_cuda:
        torch_cuda_val2 = torch.rand(1, generator=rng_suite2.torch_cuda)

    assert py_val1 == py_val2
    assert np_val1 == np_val2
    assert torch.equal(torch_cpu_val1, torch_cpu_val2)
    if torch_cuda_val1 is not None:
        assert torch_cuda_val2 is not None
        assert torch.equal(torch_cuda_val1, torch_cuda_val2)

    rng_suite3 = create_rng_suite(seed + 1)
    py_val3 = rng_suite3.python.random()
    assert py_val1 != py_val3


def test_rng_suite_independence_from_global_state() -> None:
    seed = 99
    rng_suite = create_rng_suite(seed)

    py_ref = rng_suite.python.random()
    np_ref = rng_suite.numpy.random()
    torch_ref = torch.rand(1, generator=rng_suite.torch_cpu)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1)

    py_new = rng_suite.python.random()
    np_new = rng_suite.numpy.random()
    torch_new = torch.rand(1, generator=rng_suite.torch_cpu)

    assert py_ref != py_new
    assert np_ref != np_new
    assert not torch.equal(torch_ref, torch_new)

    rng_suite_recreated = create_rng_suite(seed)
    py_recreated = rng_suite_recreated.python.random()
    np_recreated = rng_suite_recreated.numpy.random()
    torch_recreated = torch.rand(1, generator=rng_suite_recreated.torch_cpu)

    assert py_ref == py_recreated
    assert np_ref == np_recreated
    assert torch.equal(torch_ref, torch_recreated)
