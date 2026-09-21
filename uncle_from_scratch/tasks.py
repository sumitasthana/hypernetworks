"""Fixed, disjoint class tasks with task-local labels."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from data import DEFAULT_ROOT, TinyImageNet, prepare_data


DEFAULT_PARTITION = Path(__file__).parent / "task_partition.json"


def load_partition(root=DEFAULT_ROOT, path=DEFAULT_PARTITION, seed=42):
    """Create once, then verify and reuse the saved 20 x 10 class assignment.

    Sorted WordNet IDs are permuted with NumPy PCG64, independently of training
    randomness. Task IDs are zero-based. The list order defines local labels.
    A conflicting saved partition is an error, never silently overwritten.
    """
    root = prepare_data(root)
    classes = sorted((root / "wnids.txt").read_text().split())
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError("This experiment requires 200 unique Tiny ImageNet classes")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(200)
    shuffled = [classes[int(index)] for index in order]
    expected = {
        "version": 1,
        "seed": seed,
        "shuffle": "numpy.PCG64.permutation over sorted WordNet IDs",
        "task_id_base": 0,
        "classes_per_task": 10,
        "tasks": [
            {"task_id": task_id, "wnids": shuffled[start:start + 10]}
            for task_id, start in enumerate(range(0, 200, 10))
        ],
    }
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError(
                f"{path} differs from the requested partition. "
                "Use the original seed and dataset, or a new partition path."
            )
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(expected, stream, indent=2)
        stream.write("\n")
    return expected


class ClassTask(Dataset):
    """Select classes from a base dataset and translate labels to 0..K-1."""

    def __init__(self, base, wnids, task_id):
        self.base = base
        self.task_id = task_id
        self.classes = list(wnids)
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("Task classes must be nonempty and unique")
        self.class_to_idx = {wnid: local for local, wnid in enumerate(self.classes)}
        self.global_to_local = {
            base.class_to_idx[wnid]: local for wnid, local in self.class_to_idx.items()
        }
        self.class_names = [base.class_names[base.class_to_idx[w]] for w in self.classes]
        self.indices = [i for i, label in enumerate(base.targets) if label in self.global_to_local]
        self.targets = [self.global_to_local[base.targets[i]] for i in self.indices]
        if set(self.targets) != set(range(len(self.classes))):
            raise ValueError("The split is missing images for one or more task classes")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, global_label = self.base[self.indices[index]]
        return image, self.global_to_local[global_label]


def build_task(task_id=1, root=DEFAULT_ROOT, partition_path=DEFAULT_PARTITION,
               seed=42, batch_size=32):
    """Return task datasets and loaders; train shuffles, validation does not.

    Pixels remain in [0, 1], without augmentation or normalization. Worker count
    is zero for Windows/notebook compatibility. Neither split drops images.
    """
    if not isinstance(task_id, int) or not 0 <= task_id < 20:
        raise ValueError("task_id must be an integer from 0 through 19")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    partition = load_partition(root, partition_path, seed)
    wnids = partition["tasks"][task_id]["wnids"]
    train = ClassTask(TinyImageNet(root, "train", ToTensor()), wnids, task_id)
    val = ClassTask(TinyImageNet(root, "val", ToTensor()), wnids, task_id)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    return train, val, train_loader, val_loader
