from _typeshed import Incomplete
from collections.abc import Callable
from enum import StrEnum
from torch import Generator
from torch.utils.data import DataLoader, Dataset
from typing import Protocol, TypeVar

PartitionType = TypeVar('PartitionType', bound=StrEnum, contravariant=True)

class PartitionedDataset(Protocol[PartitionType]):
    def get_dataset(self, type_: PartitionType, cid: int | None) -> Dataset: ...
    def set_dataset(self, type_: PartitionType, cid: int | None, dataset: Dataset) -> None: ...
    def get_dataloader(self, type_: PartitionType, cid: int | None, batch_size: int | None, generator: Generator | None) -> DataLoader: ...

class FilteredDataset(Dataset):
    data: Incomplete
    targets: Incomplete
    transform: Incomplete
    target_transform: Incomplete
    def __init__(self, indices: list[int], original_data: list, original_targets: list | None = None, transform: Callable | None = None, target_transform: Callable | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple: ...
