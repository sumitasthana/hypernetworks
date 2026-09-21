"""The task streams: Permuted MNIST and Tiny ImageNet.

A "task" is one classification problem with `classes_per_task` labels. Permuted
MNIST makes tasks by shuffling pixels; Tiny ImageNet makes them by splitting the
200 classes into disjoint groups.
"""

import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive

from .config import Config

PIXELS = 28 * 28

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


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


class Relabelled(Dataset):
    """A subset of one class-folder dataset, with labels renumbered from 0.

    The target network has one output per class in a task, not one per class in
    the whole dataset, so global label 137 has to become local label 7.
    """

    def __init__(self, base: Dataset, indices: list[int], mapping: dict[int, int]):
        self.base = base
        self.indices = indices
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base[self.indices[index]]
        return image, self.mapping[label]


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


def _fetch_tiny_imagenet(root: Path) -> Path:
    """Download Tiny ImageNet once and put the val split into class folders.

    The official zip stores every validation image in one flat folder with a
    text file naming the class. ImageFolder needs one folder per class, so the
    first run moves the files. Later runs see the folders and skip this.
    """
    directory = root / "tiny-imagenet-200"
    if not directory.exists():
        download_and_extract_archive(TINY_IMAGENET_URL, download_root=str(root))

    flat = directory / "val" / "images"
    if flat.exists():
        annotations = (directory / "val" / "val_annotations.txt").read_text()
        for line in annotations.splitlines():
            name, wnid = line.split("\t")[:2]
            target = directory / "val" / wnid
            target.mkdir(exist_ok=True)
            shutil.move(str(flat / name), str(target / name))
        flat.rmdir()

    # ImageFolder wants the same folder layout on both sides.
    train_dir = directory / "train"
    for wnid_dir in train_dir.iterdir():
        nested = wnid_dir / "images"
        if nested.exists():
            for image in nested.iterdir():
                shutil.move(str(image), str(wnid_dir / image.name))
            nested.rmdir()

    return directory


def _class_groups(class_count: int, config: Config) -> list[list[int]]:
    """Split class indices into one disjoint group per task.

    The paper says 10 tasks of 10 classes each but never says which classes go
    together, so the rule is a config knob. "sorted" takes the classes in the
    order ImageFolder found them, which is alphabetical by wnid, and cuts them
    into consecutive blocks. "random" shuffles first, using config.seed.
    """
    order = list(range(class_count))
    if config.class_order == "random":
        generator = torch.Generator().manual_seed(config.seed)
        order = [order[index] for index in torch.randperm(class_count, generator=generator)]

    size = config.classes_per_task
    needed = len(config.tasks) * size
    if needed > class_count:
        raise ValueError(
            f"{len(config.tasks)} tasks of {size} classes needs {needed} classes, "
            f"but the dataset has {class_count}."
        )

    return [order[start:start + size] for start in range(0, needed, size)]


def _build_tiny_imagenet(config: Config) -> dict[str, dict[str, Dataset]]:
    directory = _fetch_tiny_imagenet(Path(config.data_root))
    to_tensor = transforms.ToTensor()

    train = datasets.ImageFolder(str(directory / "train"), transform=to_tensor)
    test = datasets.ImageFolder(str(directory / "val"), transform=to_tensor)

    groups = _class_groups(len(train.classes), config)

    tasks = {}
    for task, group in zip(config.tasks, groups):
        mapping = {global_label: local for local, global_label in enumerate(group)}
        tasks[task] = {
            "train": Relabelled(
                train,
                [i for i, label in enumerate(train.targets) if label in mapping],
                mapping,
            ),
            "test": Relabelled(
                test,
                [i for i, label in enumerate(test.targets) if label in mapping],
                mapping,
            ),
        }

    return tasks


def build_tasks(config: Config) -> dict[str, dict[str, Dataset]]:
    """Return {task: {"train": dataset, "test": dataset}} for every task."""
    builders = {
        "permuted_mnist": _build_permuted_mnist,
        "tiny_imagenet": _build_tiny_imagenet,
    }
    return builders[config.dataset](config)
