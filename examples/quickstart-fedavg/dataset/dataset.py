from collections.abc import Sized
from pathlib import Path

import torch
import torchvision
from blazefl.contrib import FedAvgPartitionType
from blazefl.core import FilteredDataset, PartitionedDataset
from blazefl.reproducibility import create_rng_suite
from torch import Generator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.functional import (
    balance_split,
    client_inner_dirichlet_partition_faster,
    shards_partition,
)
from dataset.transforms import GeneratorRandomCrop, GeneratorRandomHorizontalFlip


class PartitionedCIFAR10(PartitionedDataset[FedAvgPartitionType]):
    def __init__(
        self,
        root: Path,
        path: Path,
        num_clients: int,
        seed: int,
        partition: str,
        num_shards: int | None = None,
        dir_alpha: float | None = None,
    ) -> None:
        self.root = root
        self.path = path
        self.num_clients = num_clients
        self.seed = seed
        self.partition = partition
        self.num_shards = num_shards
        self.dir_alpha = dir_alpha

        self.rng_suite = create_rng_suite(seed)

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                GeneratorRandomHorizontalFlip(
                    p=0.5, generator=self.rng_suite.torch_cpu
                ),
                GeneratorRandomCrop(32, padding=4, generator=self.rng_suite.torch_cpu),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self._preprocess()

    def _preprocess(self):
        self.root.mkdir(parents=True, exist_ok=True)
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
        )
        for type_ in ["train", "test"]:
            self.path.joinpath(type_).mkdir(parents=True)

        match self.partition:
            case "client_inner_dirichlet":
                assert self.dir_alpha is not None
                client_dict = client_inner_dirichlet_partition_faster(
                    targets=train_dataset.targets,
                    num_clients=self.num_clients,
                    num_classes=10,
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        len(train_dataset.targets), self.num_clients
                    ),
                    numpy_seed=self.seed,
                )
            case "shards":
                assert self.num_shards is not None
                client_dict = shards_partition(
                    targets=train_dataset.targets,
                    num_clients=self.num_clients,
                    num_shards=self.num_shards,
                    numpy_seed=self.seed,
                )
            case _:
                raise ValueError(f"Invalid partition: {self.partition}")

        for cid, indices in client_dict.items():
            client_trainset = FilteredDataset(
                indices.tolist(),
                train_dataset.data,
                train_dataset.targets,
                transform=self.train_transform,
            )
            torch.save(client_trainset, self.path.joinpath("train", f"{cid}.pkl"))

        torch.save(
            FilteredDataset(
                list(range(len(test_dataset))),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
            ),
            self.path.joinpath("test.pkl"),
        )

    def get_dataset(self, type_: FedAvgPartitionType, cid: int | None) -> Dataset:
        match type_:
            case FedAvgPartitionType.TRAIN:
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case FedAvgPartitionType.TEST:
                dataset = torch.load(
                    self.path.joinpath(f"{type_}.pkl"), weights_only=False
                )
        assert isinstance(dataset, Dataset)
        return dataset

    def set_dataset(
        self, type_: FedAvgPartitionType, cid: int | None, dataset: Dataset
    ) -> None:
        match type_:
            case FedAvgPartitionType.TRAIN:
                torch.save(dataset, self.path.joinpath(type_, f"{cid}.pkl"))
            case FedAvgPartitionType.TEST:
                torch.save(dataset, self.path.joinpath(f"{type_}.pkl"))

    def get_dataloader(
        self,
        type_: FedAvgPartitionType,
        cid: int | None,
        batch_size: int | None = None,
        generator: Generator | None = None,
    ) -> DataLoader:
        dataset = self.get_dataset(type_, cid)
        assert isinstance(dataset, Sized)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        return data_loader
