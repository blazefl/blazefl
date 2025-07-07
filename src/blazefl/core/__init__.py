"""
Core module of BlazeFL framework.

This module imports and defines the core components of the BlazeFL framework,
including client trainers, model selectors, partitioned datasets, and server handlers.
"""

from blazefl.core.client_trainer import (
    BaseClientTrainer,
    ProcessPoolClientTrainer,
    ThreadPoolClientTrainer,
)
from blazefl.core.model_selector import ModelSelector
from blazefl.core.partitioned_dataset import FilteredDataset, PartitionedDataset
from blazefl.core.server_handler import BaseServerHandler
from blazefl.core.utils import (
    deserialize_model,
    move_tensor_to_shared_memory,
    serialize_model,
)

__all__ = [
    "BaseClientTrainer",
    "FilteredDataset",
    "ProcessPoolClientTrainer",
    "ThreadPoolClientTrainer",
    "ModelSelector",
    "PartitionedDataset",
    "BaseServerHandler",
    "serialize_model",
    "deserialize_model",
    "move_tensor_to_shared_memory",
]
