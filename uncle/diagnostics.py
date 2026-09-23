"""Observe forgetting from a saved model without retraining or copying its loop."""

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from uuid import uuid4

import torch

from . import checkpoint as checkpointing
from .config import Config
from .data import build_tasks as build_dataset_tasks
from .hypernet import HyperNetwork, build_target
from .streams import build_tasks as build_image_tasks
from .telemetry import environment
from .tinyimagenet import DEFAULT_ROOT
from .trainer import UnCLe


def diagnose_forgetting(
    checkpoint, *, task="3", forgetting_lr=None, gamma=None, steps=10,
    root=None, download=False, output=None, device=None, verbose=True, tasks=None,
):
    """Restore a trusted checkpoint and trace one forget request, starting fresh.

    No learning runs. The saved configuration defines the model and datasets;
    only forgetting_lr, gamma, and steps change the diagnostic update. Each
    call restores parameters, buffers, and RNG state and creates one optimizer
    and one fixed reference via UnCLe.forget. Every other seen task is protected,
    including previously forgotten tasks.

    Evaluation covers all seen tasks and does not consume the update RNG stream.
    Loss columns refer to before an update; accuracy columns refer to afterward.
    The initial row is step zero. This reports measurements, not a deletion test.

    output is a directory (default: a diagnostics folder beside the checkpoint).
    A uniquely named JSON report is updated after each observation. The source
    checkpoint is never written. tasks optionally supplies matching datasets
    for synthetic checks or custom data; the default builds the saved dataset.
    Tiny ImageNet uses its full validation splits, even if training was capped.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if forgetting_lr is not None and (not math.isfinite(forgetting_lr) or forgetting_lr <= 0):
        raise ValueError("forgetting_lr must be finite and positive.")
    if gamma is not None and (not math.isfinite(gamma) or gamma < 0):
        raise ValueError("gamma must be finite and nonnegative.")

    checkpoint = Path(checkpoint).expanduser().resolve()
    saved = checkpointing.load(checkpoint)
    if saved is None:
        raise FileNotFoundError(checkpoint)
    task = str(task)
    seen = list(saved["seen"])
    if task not in seen:
        raise ValueError(f"Task {task} has not been learned in this checkpoint.")
    if task in saved["forgotten"]:
        raise ValueError(f"Task {task} is already forgotten in this checkpoint.")

    training_config = Config(**saved["config"])
    rate = forgetting_lr
    if rate is None:
        rate = training_config.forgetting_learning_rate
        if rate is None:
            rate = training_config.learning_rate
    config = replace(
        training_config, forgetting_learning_rate=rate,
        gamma=training_config.gamma if gamma is None else gamma,
        device=training_config.device if device is None else str(device),
    )
    if config.torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This checkpoint uses CUDA. Enable a GPU or explicitly pass device='cpu'.")

    if tasks is None:
        if config.dataset == "tiny_imagenet":
            tasks = build_image_tasks(
                config, root=DEFAULT_ROOT if root is None else root,
                include=seen, download=download,
            )
        else:
            data_config = config if root is None else replace(config, data_root=str(root))
            tasks = build_dataset_tasks(data_config)
    for name in seen:
        if name not in tasks or "test" not in tasks[name] or len(tasks[name]["test"]) == 0:
            raise ValueError(f"Missing nonempty evaluation dataset for task {name}.")

    target = build_target(config)
    hypernet = HyperNetwork(target, config)
    uncle = UnCLe(hypernet, config, target, tasks)
    checkpointing.restore(saved, hypernet=hypernet, uncle=uncle)

    protected = [name for name in seen if name != task]
    retained = [name for name in protected if name not in saved["forgotten"]]
    destination = checkpoint.parent / "diagnostics" if output is None else Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    report_path = destination / f"forget_{stamp}_{uuid4().hex[:8]}.json"
    # Reserve the name without replacing any existing report.
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    report = {
        "checkpoint": str(checkpoint), "report_path": str(report_path.resolve()),
        "task": task, "protected": protected, "retained": retained,
        "training_config": dict(vars(training_config)),
        "settings": {"forgetting_lr": rate, "gamma": config.gamma,
                     "steps": steps, "noise_samples": config.noise_samples},
        "checkpoint_accuracies": dict(saved["previous"]),
        "environment": environment(config, hypernet, tasks),
        "status": "running", "trace": [],
    }
    del saved  # Release the CPU copy of the model after restoration.

    def write_report():
        temporary = report_path.with_suffix(".json.writing")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    def observe(record):
        # UnCLe.forget isolates this callback's RNG consumption and model modes.
        accuracies = {name: uncle.accuracy(name) for name in seen}
        initial = report["trace"][0]["accuracies"] if report["trace"] else accuracies
        row = {**record, "accuracies": accuracies,
               "target_accuracy": accuracies[task],
               "retained_drift": {name: abs(accuracies[name] - initial[name])
                                  for name in retained}}
        report["trace"].append(row)
        if record["step"] == 0:
            report["initial_matches_checkpoint"] = all(
                name in report["checkpoint_accuracies"]
                and abs(value - report["checkpoint_accuracies"][name]) < 1e-6
                for name, value in accuracies.items()
            )
        write_report()
        if verbose:
            values = "  ".join(f"{name}: {accuracies[name]:.1f}%" for name in seen)
            losses = ("" if record["step"] == 0 else
                      f"  noise={record['weighted_noise_before']:.2f}"
                      f"  preserve={record['preserve_before']:.2f}")
            print(f"step {record['step']:>3}  {values}{losses}", flush=True)
            if record["step"] == 0 and not report["initial_matches_checkpoint"]:
                print("Initial accuracies differ from the checkpoint; check evaluation data and device.")

    try:
        uncle.forget(task, protected, burn_in=steps, on_step=observe)
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_report()
        raise
    report["status"] = "complete"
    write_report()
    if verbose:
        print(f"Saved diagnostic: {report_path.resolve()}")
    return report
