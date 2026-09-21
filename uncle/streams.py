"""Turn the saved class partition into the task dict the trainer expects.

`uncle.trainer.UnCLe` wants {task_name: {"train": Dataset, "test": Dataset}}
with string names, because `Config.tasks` is a tuple of strings. This builds
that from `tasks.load_partition` and `tasks.ClassTask`.

Two things to know about the numbers this produces. Tiny ImageNet's public test
split has no labels, so the "test" slot holds the validation split: every
accuracy here is validation accuracy. And pixels stay in [0, 1] with no
normalization, which is what `uncle/data.py` does too, so the two are
comparable.
"""

from pathlib import Path

from torch.utils.data import Subset
from torchvision.transforms import ToTensor

from .tinyimagenet import DEFAULT_ROOT, TinyImageNet, prepare_data
from .tasks import DEFAULT_PARTITION, ClassTask, load_partition


def build_tasks(config=None, root=DEFAULT_ROOT, partition_path=DEFAULT_PARTITION,
                partition_seed=42, include=None, max_images=None, download=False):
    """Build the named tasks, sharing one base dataset per split.

    `tasks.build_task` rebuilds the base dataset per call and returns loaders
    we would throw away, so this goes to `ClassTask` directly.

    `include` names which tasks to build, defaulting to `config.tasks`, then to
    all of them. Building a task indexes both splits, so a short run should
    only ask for the tasks its requests mention. `max_images` caps images per
    task, for fast checks. `download` fetches the archive when `root` is
    missing, which is what a fresh Colab runtime needs. `partition_seed` is the
    seed of the saved partition file, not the training seed, so it stays 42
    unless the file moves too.
    """
    # Resolve once, with the download here rather than in every caller below.
    # After this, load_partition and TinyImageNet find an existing directory.
    root = prepare_data(root, download=download)

    if include is None:
        include = config.tasks if config is not None else None
    wanted = None if include is None else {str(name) for name in include}

    partition = load_partition(root, Path(partition_path), partition_seed)
    groups = [
        group for group in partition["tasks"]
        if wanted is None or str(group["task_id"]) in wanted
    ]
    if wanted is not None and len(groups) != len(wanted):
        missing = wanted - {str(group["task_id"]) for group in groups}
        raise ValueError(f"The partition has no task named {sorted(missing)}")

    # Only the classes these tasks name. Indexing all 200 to use 40 of them
    # was the largest cost before the first optimizer step.
    wnids = [wnid for group in groups for wnid in group["wnids"]]
    bases = {
        slot: TinyImageNet(root, split, ToTensor(), classes=wnids)
        for slot, split in (("train", "train"), ("test", "val"))
    }
    return {
        str(group["task_id"]): {
            slot: _capped(ClassTask(base, group["wnids"], group["task_id"]), max_images)
            for slot, base in bases.items()
        }
        for group in groups
    }


def _capped(task, max_images):
    """Keep the first `max_images` of every class, so all ten stay represented."""
    if max_images is None or len(task) <= max_images:
        return task
    per_class = max(1, max_images // len(task.classes))
    kept, counts = [], {}
    for index, label in enumerate(task.targets):
        if counts.get(label, 0) < per_class:
            counts[label] = counts.get(label, 0) + 1
            kept.append(index)
    return Subset(task, kept)
