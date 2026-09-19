"""Permuted MNIST: every task is the same digits with its pixels shuffled."""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .config import Config

PIXELS = 28 * 28


class Permuted(Dataset):
    """Wraps MNIST and rearranges each image by one fixed shuffle."""

    def __init__(self, base: Dataset, permutation: torch.Tensor):
        self.base = base
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image.reshape(-1)[self.permutation].reshape(1, 28, 28), label


def build_tasks(config: Config) -> dict[str, dict[str, Dataset]]:
    """Return {task: {"train": dataset, "test": dataset}} for every task."""
    to_tensor = transforms.ToTensor()
    train = datasets.MNIST(
        config.data_root, train=True, download=True, transform=to_tensor
    )
    test = datasets.MNIST(
        config.data_root, train=False, download=True, transform=to_tensor
    )

    # One generator for the shuffles, so they do not depend on training randomness.
    generator = torch.Generator().manual_seed(config.seed)

    tasks = {}
    for task in config.tasks:
        permutation = torch.randperm(PIXELS, generator=generator)
        tasks[task] = {
            "train": Permuted(train, permutation),
            "test": Permuted(test, permutation),
        }

    return tasks
