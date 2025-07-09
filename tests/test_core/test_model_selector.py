from enum import StrEnum

import torch

from src.blazefl.core import ModelSelector


class DummyModelName(StrEnum):
    LINEAR = "linear"
    CONV = "conv"


class DummyModelSelector(ModelSelector[DummyModelName]):
    def select_model(self, model_name: DummyModelName) -> torch.nn.Module:
        match model_name:
            case DummyModelName.LINEAR:
                return torch.nn.Linear(10, 5)
            case DummyModelName.CONV:
                return torch.nn.Conv2d(1, 3, 3)


def test_model_selector_subclass() -> None:
    selector = DummyModelSelector()
    model = selector.select_model(DummyModelName.LINEAR)
    assert isinstance(model, torch.nn.Linear)
    model = selector.select_model(DummyModelName.CONV)
    assert isinstance(model, torch.nn.Conv2d)
