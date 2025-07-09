from enum import StrEnum

import torch
from blazefl.core import ModelSelector
from torch import nn
from torchvision.models import resnet18

from models.cnn import CNN


class DSFLModelName(StrEnum):
    CNN = "cnn"
    RESNET18 = "resnet18"


class DSFLModelSelector(ModelSelector[DSFLModelName]):
    def __init__(self, num_classes: int, seed: int) -> None:
        self.num_classes = num_classes
        self.seed = seed

    def select_model(self, model_name: DSFLModelName) -> nn.Module:
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            match model_name:
                case DSFLModelName.CNN:
                    return CNN(num_classes=self.num_classes)
                case DSFLModelName.RESNET18:
                    return resnet18(num_classes=self.num_classes)
