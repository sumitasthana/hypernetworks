"""Train the target network on one task the ordinary way, as a sanity check.

Nothing here involves a hypernetwork. It answers one question: can this
architecture learn a single Tiny ImageNet task at all? If this cannot beat 10%
on ten classes, no hypernetwork result downstream is worth debugging.

The network comes from `uncle.hypernet.build_target`, so it is the same
architecture the hypernetwork later generates weights for.

From a notebook:

    from baseline import run_baseline
    report = run_baseline(task=1, backbone="resnet50", epochs=10)
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, dataset_defaults
from .hypernet import build_target
from .streams import build_tasks
from .tinyimagenet import DEFAULT_ROOT


def evaluate(network, loader, device) -> float:
    network.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            scores = network(images.to(device))
            correct += (scores.argmax(1) == labels.to(device)).sum().item()
            total += labels.numel()
    return 100.0 * correct / total


def train(network, train_loader, val_loader, device, epochs, learning_rate,
          verbose=True) -> list[float]:
    """Plain supervised training. Returns validation accuracy after each epoch."""
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    accuracies = []
    for epoch in range(epochs):
        network.train()
        for images, labels in train_loader:
            loss = F.cross_entropy(network(images.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        accuracies.append(evaluate(network, val_loader, device))
        if verbose:
            print(f"epoch {epoch + 1}: validation accuracy {accuracies[-1]:.2f}%",
                  flush=True)
    return accuracies


def run_baseline(task=1, epochs=5, learning_rate=1e-3, max_images=None,
                 root=DEFAULT_ROOT, download=False, output=None, verbose=True,
                 **overrides) -> dict:
    """Train one task directly and return a report dict.

    Any `Config` field can be passed as a keyword argument: `backbone`,
    `batch_size`, `seed`, `device`, and so on. `output` is a directory for the
    JSON report, or None to just return it. `download` fetches the dataset when
    it is missing, which a fresh Colab runtime needs.
    """
    config = Config(**{
        **dataset_defaults("tiny_imagenet", 1),
        "backbone": "cnn",
        **overrides,
    })
    torch.manual_seed(config.seed)

    # Only this task, because building one indexes both splits.
    name = str(task)
    tasks = build_tasks(config, root=root, include=[name],
                        max_images=max_images, download=download)

    # build_target freezes the network, because the hypernetwork usually
    # supplies its weights. Here we train them, so switch gradients back on.
    network = build_target(config).requires_grad_(True)
    train_loader = DataLoader(
        tasks[name]["train"], batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(tasks[name]["test"], batch_size=config.eval_batch_size)

    if verbose:
        print(f"task {name}: {len(train_loader.dataset)} train, "
              f"{len(val_loader.dataset)} validation, backbone {config.backbone}, "
              f"device {config.device}")

    accuracies = train(network, train_loader, val_loader, config.torch_device,
                       epochs, learning_rate, verbose=verbose)

    report = {
        "task": name,
        "backbone": config.backbone,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "max_images": max_images,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "validation_accuracy_per_epoch": accuracies,
        "best_validation_accuracy": max(accuracies) if accuracies else None,
        "chance_accuracy": 100.0 / config.classes_per_task,
    }

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"task{name}.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if verbose:
            print(f"Saved baseline report to {path.resolve()}")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--backbone", default="cnn",
                        choices=("cnn", "resnet18", "resnet50"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "baseline")
    args = parser.parse_args()

    overrides = {"device": args.device} if args.device else {}
    run_baseline(
        task=args.task,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_images=args.max_images,
        root=args.root,
        download=args.download,
        output=args.output,
        backbone=args.backbone,
        batch_size=args.batch_size,
        seed=args.seed,
        **overrides,
    )


if __name__ == "__main__":
    main()
