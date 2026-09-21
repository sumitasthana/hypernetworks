"""The task streams: Permuted MNIST here, Tiny ImageNet in streams.py.

A "task" is one classification problem with `classes_per_task` labels. Permuted
MNIST makes tasks by shuffling pixels. Tiny ImageNet makes them by splitting the
200 classes into disjoint groups, which `streams.build_tasks` does from the
saved partition; this module just routes to it, so there is one loader rather
than two disagreeing about which classes make up task 3.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .config import Config
from .streams import build_tasks as build_tiny_imagenet_tasks

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


def _build_permuted_mnist(config: Config) -> dict[str, dict[str, Dataset]]:
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


def build_tasks(config: Config) -> dict[str, dict[str, Dataset]]:
    """Return {task: {"train": dataset, "test": dataset}} for every task."""
    if config.dataset == "tiny_imagenet":
        # One Tiny ImageNet loader for the whole project, in streams.py. It
        # reads the archive where it lies and takes its tasks from the saved,
        # seed-42 partition, which is checked on every run.
        return build_tiny_imagenet_tasks(
            config,
            root=Path(config.data_root) / "tiny-imagenet-200",
            download=True,
        )
    return _build_permuted_mnist(config)
