import torch
from enum import StrEnum
from typing import Protocol, TypeVar

ModelName = TypeVar('ModelName', bound=StrEnum, contravariant=True)

class ModelSelector(Protocol[ModelName]):
    def select_model(self, model_name: ModelName) -> torch.nn.Module: ...
