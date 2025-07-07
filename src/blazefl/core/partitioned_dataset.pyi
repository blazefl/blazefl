from enum import StrEnum
from torch import Generator
from torch.utils.data import DataLoader, Dataset
from typing import Protocol, TypeVar

PartitionType = TypeVar('PartitionType', bound=StrEnum, contravariant=True)

class PartitionedDataset(Protocol[PartitionType]):
    def get_dataset(self, type_: PartitionType, cid: int | None) -> Dataset: ...
    def set_dataset(self, type_: PartitionType, cid: int | None, dataset: Dataset) -> None: ...
    def get_dataloader(self, type_: PartitionType, cid: int | None, batch_size: int | None, generator: Generator | None) -> DataLoader: ...
