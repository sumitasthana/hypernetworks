"""Run a Tiny ImageNet learn-and-unlearn sequence and report the four numbers.

Written to be called, not just run. From a notebook:

    from run import run_experiment
    history, numbers = run_experiment(sequence=1, backbone="resnet50", epochs=5)

Any `Config` field can be passed as a keyword argument, so sweeping a
hyperparameter is a loop over calls:

    for gamma in (0.001, 0.01, 0.1):
        run_experiment(backbone="cnn", chunks=32, epochs=1, gamma=gamma)

The tasks come from this folder (`streams.build_tasks`, on the seed-42
partition). The method comes from the sibling package: `uncle.hypernet`,
`uncle.trainer`, `uncle.metrics`, and the paper's Table 4 request sequences in
`uncle.config`.
"""

import argparse
import json
from pathlib import Path
import sys

# Both this folder and the repository root, so the flat imports below and the
# `uncle` package both resolve wherever the caller started from. A Colab cell
# only has to put this folder on the path once.
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(_HERE.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from uncle.config import Config, dataset_defaults
from uncle.experiment import run as run_requests
from uncle.metrics import summary

from data import DEFAULT_ROOT
from streams import build_tasks


def make_config(sequence=1, limit_requests=None, **overrides) -> Config:
    """The paper's Tiny ImageNet setting for one row of Table 4.

    `dataset_defaults` supplies the task list, the request sequence and beta.
    Anything else is a keyword argument. Kept separate from `run_experiment` so
    a notebook can print or adjust the config before spending GPU hours on it.
    """
    defaults = dataset_defaults("tiny_imagenet", sequence)
    if limit_requests is not None:
        # Cutting from the front keeps the list valid: every forget still
        # follows its learn. Config checks that anyway.
        defaults["requests"] = defaults["requests"][:limit_requests]
    return Config(**{**defaults, **overrides})


def describe(record) -> str:
    """One line per finished request."""
    accuracies = " ".join(
        f"{task}:{record['after'][task]:.1f}" for task in record["seen"]
    )
    burn_in = "" if record["burn_in"] is None else f" burn_in={record['burn_in']}"
    return (f"[{record['index']:>2}] {record['action']:<6} task {record['task']:<2} "
            f"loss={record['final_loss']:.4f}{burn_in}  {accuracies}")


def show(value) -> str:
    """The metrics return None when a run has no forget request yet."""
    return "n/a" if value is None else f"{value:.2f}"


def run_experiment(sequence=1, limit_requests=None, max_images=None,
                   root=DEFAULT_ROOT, download=False, output=None,
                   on_request=None, verbose=True, config=None, **overrides):
    """Work through a request sequence. Returns (history, four numbers).

    `output` is a directory for history.json and summary.json, or None to keep
    everything in memory, which is usually what a notebook wants. When it is
    set, the history is rewritten after every request, so a thirty-request
    ResNet50 run can be watched and survives a crash on request 25.

    `on_request` is called with each record as it completes, on top of the
    printing `verbose` does, so a notebook can plot as the run goes. `config`
    takes a prepared Config and ignores the sequence and override arguments.
    `download` fetches the dataset when it is missing, which a fresh Colab
    runtime needs.
    """
    if config is None:
        config = make_config(sequence, limit_requests, **overrides)
    elif overrides:
        raise ValueError("Pass overrides to make_config, or a finished config, not both")

    # Only the tasks these requests mention, because building one indexes both
    # splits. A full sequence names all twenty anyway.
    needed = sorted({task for _, task in config.requests}, key=int)
    tasks = build_tasks(config, root=root, include=needed,
                        max_images=max_images, download=download)

    history_path = summary_path = None
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        history_path = output / f"history_sequence{sequence}.json"
        summary_path = output / f"summary_sequence{sequence}.json"

    if verbose:
        print(f"sequence {sequence}: {len(config.requests)} requests over "
              f"{len(needed)} tasks, backbone {config.backbone}, "
              f"{config.chunks} chunks, beta {config.beta}, gamma {config.gamma}, "
              f"device {config.device}")

    history = []

    def record_done(record):
        history.append(record)
        if verbose:
            print(describe(record), flush=True)
        if history_path is not None:
            history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if on_request is not None:
            on_request(record)

    run_requests(config, on_request=record_done, tasks=tasks)

    numbers = summary(history)
    if summary_path is not None:
        summary_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")

    if verbose:
        chance = 100 / config.classes_per_task
        print()
        print(f"retain accuracy  {show(numbers['retain_accuracy'])}%")
        print(f"forget accuracy  {show(numbers['forget_accuracy'])}%  "
              f"(chance is {chance:.2f}%)")
        print(f"mean spill       {show(numbers['mean_spill'])}")
        print(f"mean relapse     {show(numbers['mean_relapse'])}")
        if output is not None:
            print(f"\nSaved {history_path.name} and {summary_path.name} "
                  f"to {output.resolve()}")

    return history, numbers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=1, choices=(1, 2, 3),
                        help="which row of the paper's Table 4")
    parser.add_argument("--backbone", default="resnet50",
                        choices=("cnn", "resnet18", "resnet50"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=None,
                        help="default follows the dataset: 0.01 for Tiny ImageNet")
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--limit-requests", type=int, default=None,
                        help="stop after this many requests, for a short check")
    parser.add_argument("--max-images", type=int, default=None,
                        help="cap images per task, for a short check")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=_HERE / "outputs" / "run")
    args = parser.parse_args()

    overrides = {
        name: value
        for name, value in (("beta", args.beta), ("device", args.device))
        if value is not None
    }
    run_experiment(
        sequence=args.sequence,
        limit_requests=args.limit_requests,
        max_images=args.max_images,
        root=args.root,
        download=args.download,
        output=args.output,
        backbone=args.backbone,
        epochs=args.epochs,
        chunks=args.chunks,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        seed=args.seed,
        **overrides,
    )


if __name__ == "__main__":
    main()
