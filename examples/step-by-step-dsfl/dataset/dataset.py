from collections.abc import Sized
from enum import StrEnum
from pathlib import Path

import torch
import torchvision
from blazefl.core import FilteredDataset, PartitionedDataset
from blazefl.reproducibility import create_rng_suite
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .functional import balance_split, client_inner_dirichlet_partition_faster


class DSFLPartitionType(StrEnum):
    TRAIN = "train"
    OPEN = "open"
    TEST = "test"


class DSFLPartitionedDataset(PartitionedDataset[DSFLPartitionType]):
    def __init__(
        self,
        root: Path,
        path: Path,
        num_clients: int,
        seed: int,
        partition: str,
        open_size: int,
        dir_alpha: float,
    ) -> None:
        self.root = root
        self.path = path
        self.num_clients = num_clients
        self.seed = seed
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.open_size = open_size

        self.rng_suite = create_rng_suite(seed)

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
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
        open_dataset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
        )
        for type_ in [ds.value for ds in DSFLPartitionType]:
            self.path.joinpath(type_).mkdir(parents=True)

        match self.partition:
            case "client_inner_dirichlet":
                client_dict, class_priors = client_inner_dirichlet_partition_faster(
                    targets=train_dataset.targets,
                    num_clients=self.num_clients,
                    num_classes=10,
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        num_clients=self.num_clients,
                        num_samples=len(train_dataset.targets),
                    ),
                    verbose=False,
                    numpy_rng=self.rng_suite.numpy,
                )
                test_client_dict, _ = client_inner_dirichlet_partition_faster(
                    targets=test_dataset.targets,
                    num_clients=self.num_clients,
                    num_classes=10,
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        num_clients=self.num_clients,
                        num_samples=len(test_dataset.targets),
                    ),
                    class_priors=class_priors,
                    verbose=False,
                    numpy_rng=self.rng_suite.numpy,
                )
            case _:
                raise ValueError(f"Invalid partition: {self.partition}")

        for cid, indices in client_dict.items():
            client_train_dataset = FilteredDataset(
                indices.tolist(),
                train_dataset.data,
                train_dataset.targets,
                transform=self.train_transform,
            )
            torch.save(client_train_dataset, self.path.joinpath("train", f"{cid}.pkl"))

        for cid, indices in test_client_dict.items():
            client_test_dataset = FilteredDataset(
                indices.tolist(),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
            )
            torch.save(client_test_dataset, self.path.joinpath("test", f"{cid}.pkl"))

        open_indices = self.rng_suite.numpy.choice(
            len(open_dataset),
            size=self.open_size,
        )
        torch.save(
            FilteredDataset(
                open_indices.tolist(),
                open_dataset.data,
                original_targets=None,
                transform=self.train_transform,
            ),
            self.path.joinpath("open.pkl"),
        )

        torch.save(
            FilteredDataset(
                list(range(len(test_dataset))),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
            ),
            self.path.joinpath("test", "default.pkl"),
        )

    def get_dataset(self, type_: DSFLPartitionType, cid: int | None) -> Dataset:
        match type_:
            case DSFLPartitionType.TRAIN:
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case DSFLPartitionType.OPEN:
                dataset = torch.load(
                    self.path.joinpath(f"{type_}.pkl"),
                    weights_only=False,
                )
            case DSFLPartitionType.TEST:
                if cid is not None:
                    dataset = torch.load(
                        self.path.joinpath(type_, f"{cid}.pkl"),
                        weights_only=False,
                    )
                else:
                    dataset = torch.load(
                        self.path.joinpath(type_, "default.pkl"), weights_only=False
                    )
        assert isinstance(dataset, Dataset)
        return dataset

    def set_dataset(
        self, type_: DSFLPartitionType, cid: int | None, dataset: Dataset
    ) -> None:
        match type_:
            case DSFLPartitionType.TRAIN:
                torch.save(dataset, self.path.joinpath(type_, f"{cid}.pkl"))
            case DSFLPartitionType.OPEN:
                torch.save(dataset, self.path.joinpath(f"{type_}.pkl"))
            case DSFLPartitionType.TEST:
                if cid is not None:
                    torch.save(dataset, self.path.joinpath(type_, f"{cid}.pkl"))
                else:
                    torch.save(dataset, self.path.joinpath(type_, "default.pkl"))

    def get_dataloader(
        self,
        type_: DSFLPartitionType,
        cid: int | None,
        batch_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        dataset = self.get_dataset(type_, cid)
        assert isinstance(dataset, Sized)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=type_ == DSFLPartitionType.TRAIN,
            generator=generator,
        )
        return data_loader
