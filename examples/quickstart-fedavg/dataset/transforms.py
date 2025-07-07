import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import RandomCrop, RandomHorizontalFlip


class GeneratorRandomCrop(RandomCrop):
    def __init__(
        self,
        size,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
        generator: torch.Generator | None = None,
    ):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.generator = generator

    def _get_params(
        self, img: Tensor, output_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        i, j, h, w = super().get_params(img, output_size)

        th, tw = output_size
        i = torch.randint(0, h - th + 1, size=(1,), generator=self.generator).item()
        j = torch.randint(0, w - tw + 1, size=(1,), generator=self.generator).item()

        return int(i), int(j), th, tw

    def forward(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        assert len(self.size) == 2
        i, j, h, w = self._get_params(img, self.size)

        return F.crop(img, i, j, h, w)


class GeneratorRandomHorizontalFlip(RandomHorizontalFlip):
    def __init__(self, p=0.5, generator: torch.Generator | None = None):
        super().__init__(p)
        self.generator = generator

    def forward(self, img):
        if torch.rand(1, generator=self.generator) < self.p:
            return F.hflip(img)
        return img
