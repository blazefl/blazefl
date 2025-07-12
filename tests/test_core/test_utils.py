from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from src.blazefl.core import (
    SHMHandle,
    deserialize_model,
    process_tensors_in_object,
    reconstruct_from_shared_memory,
    serialize_model,
)


@pytest.fixture
def simple_model() -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1)
    )
    return model


def test_serialize_deserialize_cpu(simple_model: torch.nn.Module) -> None:
    original_params = [p.clone() for p in simple_model.parameters()]
    serialized = serialize_model(simple_model, cpu=True)

    assert serialized.device.type == "cpu"

    total_numel = sum(p.numel() for p in simple_model.parameters())
    assert serialized.numel() == total_numel

    for p in simple_model.parameters():
        p.data.normal_()

    deserialize_model(simple_model, serialized)

    for orig_p, new_p in zip(original_params, simple_model.parameters(), strict=True):
        assert torch.allclose(orig_p, new_p), "Parameters did not restore correctly."


def test_serialize_deserialize_gpu(simple_model: torch.nn.Module) -> None:
    if torch.cuda.is_available():
        simple_model = simple_model.to("cuda")
        original_params = [p.clone() for p in simple_model.parameters()]

        serialized_gpu = serialize_model(simple_model, cpu=False)
        assert serialized_gpu.device.type == "cuda"

        serialized_cpu = serialize_model(simple_model, cpu=True)
        assert serialized_cpu.device.type == "cpu"

        for p in simple_model.parameters():
            p.data.normal_()

        simple_model = simple_model.cpu()
        deserialize_model(simple_model, serialized_cpu)

        for orig_p, new_p in zip(
            original_params, simple_model.parameters(), strict=True
        ):
            assert torch.allclose(
                orig_p.cpu(), new_p
            ), "Parameters did not restore correctly."


@dataclass
class NestedData:
    tensor: torch.Tensor
    id: int


@dataclass
class ComplexData:
    tensor_a: torch.Tensor
    tensor_b: torch.Tensor
    metadata: dict[str, Any]
    nested: NestedData
    # Field for creating circular references
    parent: Any = field(default=None, repr=False)


@pytest.fixture
def complex_object() -> ComplexData:
    """Creates a nested dataclass object for testing."""
    obj = ComplexData(
        tensor_a=torch.randn(2, 2),
        tensor_b=torch.ones(5),
        metadata={"name": "test_object", "value": 123},
        nested=NestedData(tensor=torch.arange(4), id=1),
    )
    return obj


@pytest.fixture
def circular_object() -> ComplexData:
    """Creates an object with a circular reference."""
    obj = ComplexData(
        tensor_a=torch.randn(2, 2),
        tensor_b=torch.ones(5),
        metadata={"name": "circular_test"},
        nested=NestedData(tensor=torch.arange(4), id=2),
    )
    # Create the circular reference
    obj.parent = obj
    return obj


def test_process_tensors_move_mode(complex_object: ComplexData):
    """
    Tests that 'move' mode modifies tensors in-place and doesn't change the object ID.
    """
    original_id = id(complex_object)
    # Mock share_memory_ to check if it's called
    call_count = 0
    original_share_memory = torch.Tensor.share_memory_

    def mocked_share_memory(self):
        nonlocal call_count
        call_count += 1

    torch.Tensor.share_memory_ = mocked_share_memory

    result = process_tensors_in_object(complex_object, mode="move", max_depth=10)

    # Restore original method
    torch.Tensor.share_memory_ = original_share_memory

    assert id(result) == original_id, "Object ID should be the same in 'move' mode"
    assert call_count == 3, "share_memory_() should be called on all 3 tensors"


def test_process_tensors_replace_mode(complex_object: ComplexData):
    """
    Tests that 'replace' mode creates a new object and replaces tensors with SHMHandle.
    """
    original_id = id(complex_object)
    handle_package = process_tensors_in_object(
        complex_object, mode="replace", max_depth=10
    )

    assert (
        id(handle_package) != original_id
    ), "Object ID should be different in 'replace' mode"
    assert isinstance(handle_package.tensor_a, SHMHandle)
    assert isinstance(handle_package.tensor_b, SHMHandle)
    assert isinstance(handle_package.nested.tensor, SHMHandle)
    assert (
        handle_package.metadata["name"] == "test_object"
    ), "Non-tensor data should be preserved"


def test_reconstruct_from_shared_memory(complex_object: ComplexData):
    """
    Tests the reconstruction logic, ensuring non-tensor data from the handle object
    and tensor data from the shm_buffer are correctly combined.
    """
    # 1. Create the shared memory buffer (pretending it's in SHM)
    shm_buffer = complex_object

    # 2. Create the handle package
    handle_package = process_tensors_in_object(shm_buffer, mode="replace", max_depth=10)

    # 3. Modify a non-tensor value in the handle package (simulating a worker update)
    handle_package.metadata["value"] = 999
    handle_package.nested.id = 5

    # 4. Reconstruct the final object
    final_package = reconstruct_from_shared_memory(handle_package, shm_buffer)

    # 5. Assertions
    assert torch.allclose(
        final_package.tensor_a, shm_buffer.tensor_a
    ), "Tensor A should be from shm_buffer"
    assert torch.allclose(
        final_package.nested.tensor, shm_buffer.nested.tensor
    ), "Nested tensor should be from shm_buffer"
    assert (
        final_package.metadata["value"] == 999
    ), "Metadata update from handle_package should be preserved"
    assert (
        final_package.nested.id == 5
    ), "Nested ID update from handle_package should be preserved"
    assert not isinstance(
        final_package.tensor_a, SHMHandle
    ), "Final package should not contain handles"


def test_circular_reference_handling(circular_object: ComplexData):
    """
    Tests that both functions complete without infinite recursion when a circular
    reference is present.
    """
    # Test 'move' mode
    try:
        process_tensors_in_object(circular_object, mode="move", max_depth=10)
    except RecursionError:
        pytest.fail(
            "process_tensors_in_object in 'move' mode failed on circular reference"
        )

    # Test 'replace' mode
    try:
        handle_package = process_tensors_in_object(
            circular_object, mode="replace", max_depth=10
        )
    except RecursionError:
        pytest.fail(
            "process_tensors_in_object in 'replace' mode failed on circular reference"
        )

    # Test reconstruction
    try:
        reconstruct_from_shared_memory(handle_package, circular_object)
    except RecursionError:
        pytest.fail("reconstruct_from_shared_memory failed on circular reference")


def test_max_depth_parameter(complex_object: ComplexData):
    """
    Tests that recursion stops at the specified max_depth.
    """
    # With max_depth=1, only top-level tensors should be replaced.
    handle_package = process_tensors_in_object(
        complex_object, mode="replace", max_depth=2
    )  # depth starts at 0, __dict__ is 1

    assert isinstance(handle_package.tensor_a, SHMHandle)
    # The nested tensor should NOT be a handle, as it's beyond max_depth
    assert isinstance(handle_package.nested.tensor, torch.Tensor)
