from typing import Any

import torch


def move_tensor_to_shared_memory(obj: Any, max_depth: int = 1) -> None:
    visited = set()

    def _recursive_helper(current_obj: Any, depth: int):
        if depth >= max_depth:
            return

        if isinstance(current_obj, dict | list | tuple) or hasattr(
            current_obj, "__dict__"
        ):
            obj_id = id(current_obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

        if isinstance(current_obj, torch.Tensor):
            current_obj.share_memory_()
        elif isinstance(current_obj, dict):
            for v in current_obj.values():
                _recursive_helper(v, depth + 1)
        elif isinstance(current_obj, list | tuple):
            for item in current_obj:
                _recursive_helper(item, depth + 1)
        elif hasattr(current_obj, "__dict__"):
            for v in current_obj.__dict__.values():
                _recursive_helper(v, depth + 1)

    _recursive_helper(obj, 0)


def serialize_model(model: torch.nn.Module, cpu: bool = True) -> torch.Tensor:
    """
    Serialize a PyTorch model's parameters into a flat tensor.

    Args:
        model (torch.nn.Module): The PyTorch model to serialize.
        cpu (bool): Whether to move the serialized parameters to the CPU.

    Returns:
        torch.Tensor: A flat tensor containing the serialized parameters.
    """
    parameters = [param.data.view(-1) for param in model.state_dict().values()]
    serialized_parameters = torch.cat(parameters)
    if cpu:
        serialized_parameters = serialized_parameters.cpu()

    return serialized_parameters


def deserialize_model(
    model: torch.nn.Module, serialized_parameters: torch.Tensor
) -> None:
    """
    Deserialize a flat tensor back into a PyTorch model's parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to update.
        serialized_parameters (torch.Tensor): The tensor containing the parameters.

    Returns:
        None
    """
    current_index = 0
    for param in model.state_dict().values():
        numel = param.numel()
        size = param.size()
        param.copy_(
            serialized_parameters[current_index : current_index + numel].view(size)
        )
        current_index += numel
