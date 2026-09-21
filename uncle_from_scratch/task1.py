"""Prepare and inspect task ID 1. Run from any working directory."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import DEFAULT_ROOT
from tasks import DEFAULT_PARTITION, build_task


def describe_task(train, val):
    """Summarize the actual split counts and the meaning of each local label."""
    if train.class_to_idx != val.class_to_idx:
        raise ValueError("Training and validation label mappings differ")
    train_counts, val_counts = Counter(train.targets), Counter(val.targets)
    return {
        "task_id": train.task_id,
        "task_id_base": 0,
        "num_classes": len(train.classes),
        "train_images": len(train),
        "val_images": len(val),
        "classes": [
            {"local_label": local, "global_label": train.base.class_to_idx[wnid],
             "wnid": wnid, "name": train.class_names[local],
             "train_images": train_counts[local], "val_images": val_counts[local]}
            for local, wnid in enumerate(train.classes)
        ],
    }


def show_task(train, seed=42):
    """Display one tensor image from each task class, labeled 0 through 9."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(train.targets)
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    for label, ax in enumerate(axes.flat):
        index = int(rng.choice(np.flatnonzero(targets == label)))
        image, actual_label = train[index]
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(f"{actual_label}: {train.class_names[label].split(',')[0]}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Task ID {train.task_id}: ten classes with local labels")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "task1")
    args = parser.parse_args()
    train, val, train_loader, val_loader = build_task(
        root=args.root, partition_path=args.partition, batch_size=args.batch_size,
    )
    report = describe_task(train, val)
    report["partition_file"] = str(args.partition.resolve())
    report["batches"] = {}
    for name, loader in (("train", train_loader), ("val", val_loader)):
        images, labels = next(iter(loader))
        report["batches"][name] = {
            "images_shape": list(images.shape), "labels_shape": list(labels.shape),
            "images_dtype": str(images.dtype), "labels_dtype": str(labels.dtype),
            "pixel_range": [images.min().item(), images.max().item()],
        }
    args.output.mkdir(parents=True, exist_ok=True)
    fig = show_task(train)
    fig.savefig(args.output / "examples.png", dpi=140)
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved task 1 outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
