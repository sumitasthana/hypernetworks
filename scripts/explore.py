"""Explore real images and save a small, reproducible report. No training yet."""

import argparse
from pathlib import Path as _Path
import sys

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from uncle.tinyimagenet import DEFAULT_ROOT, TinyImageNet, prepare_data


def class_counts(dataset):
    return np.bincount(dataset.targets, minlength=len(dataset.classes))


def show_examples(dataset, seed=42):
    """Show one image from each of 16 randomly selected classes."""
    rng = np.random.default_rng(seed)
    classes = rng.choice(len(dataset.classes), min(16, len(dataset.classes)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    targets = np.asarray(dataset.targets)
    for ax in axes.flat:
        ax.axis("off")
    for ax, label in zip(axes.flat, classes):
        index = int(rng.choice(np.flatnonzero(targets == label)))
        image, _ = dataset[index]
        ax.imshow(image)
        ax.set_title(f"{label}: {dataset.class_names[label].split(',')[0]}", fontsize=9)
    fig.suptitle(f"Tiny ImageNet: {dataset.split} examples (seed {seed})")
    fig.tight_layout()
    return fig


def pixel_statistics(dataset, sample_size=2048, seed=42):
    """Estimate RGB population mean/std on sampled training pixels in [0, 1].

    Read raw modes before RGB conversion. Accumulate pixels rather than
    averaging per-image standard deviations. This is not a full corruption scan.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    indices = np.random.default_rng(seed).choice(
        len(dataset), min(sample_size, len(dataset)), replace=False
    )
    total = np.zeros(3, dtype=np.float64)
    squared = np.zeros(3, dtype=np.float64)
    pixels = 0
    modes, sizes = Counter(), Counter()
    for index in indices:
        path = dataset.image_path(int(index))
        with Image.open(path) as image:
            modes[image.mode] += 1
            sizes[f"{image.width}x{image.height}"] += 1
            values = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255
        total += values.sum(axis=0)
        squared += (values * values).sum(axis=0)
        pixels += len(values)
    mean = total / pixels
    std = np.sqrt(np.maximum(squared / pixels - mean * mean, 0))
    return {"sample_images": len(indices), "seed": seed,
            "raw_modes": dict(modes), "sizes": dict(sizes),
            "rgb_mean": mean.tolist(), "rgb_std": std.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    root = prepare_data(args.root, args.download)
    train, val = TinyImageNet(root, "train"), TinyImageNet(root, "val")
    assert train.class_to_idx == val.class_to_idx
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {"train": class_counts(train), "val": class_counts(val)}
    fig = show_examples(train, args.seed)
    fig.savefig(args.output / "train_examples.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (split, values) in zip(axes, counts.items()):
        ax.bar(np.arange(len(values)), values)
        ax.set_ylabel(f"{split} images")
    axes[-1].set_xlabel("Class index (sorted WordNet ID)")
    fig.tight_layout()
    fig.savefig(args.output / "class_balance.png", dpi=140)
    plt.close(fig)
    tensors = TinyImageNet(root, "train", transform=ToTensor())
    loader = DataLoader(tensors, batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    images, labels = next(iter(loader))
    report = {
        "dataset_root": str(root), "classes": len(train.classes),
        "split_images": {"train": len(train), "val": len(val)},
        "images_per_class": {split: {"min": int(v.min()), "max": int(v.max())}
                             for split, v in counts.items()},
        "training_pixel_sample": pixel_statistics(train, args.sample_size, args.seed),
        "batch": {"images_shape": list(images.shape), "labels_shape": list(labels.shape),
                  "images_dtype": str(images.dtype), "labels_dtype": str(labels.dtype),
                  "pixel_min": images.min().item(), "pixel_max": images.max().item()},
        "note": "Pixel statistics are sampled, not a full image integrity audit. Test labels are unavailable.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved figures and summary to {args.output.resolve()}")


if __name__ == "__main__":
    main()
