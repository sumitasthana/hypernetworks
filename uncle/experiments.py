"""Run Tiny ImageNet learn-and-unlearn sequences, and record what they cost.

Written to be called, not just run. From a notebook:

    from uncle.experiments import run_experiment
    result = run_experiment(sequence=1, backbone="resnet50", epochs=5)
    print(result["numbers"], result["totals"])

Any `Config` field can be passed as a keyword argument, so sweeping a
hyperparameter is a loop over calls:

    for gamma in (0.001, 0.01, 0.1):
        run_experiment(backbone="cnn", chunks=32, epochs=1, gamma=gamma)

All three of the paper's sequences, over several seeds, in one call:

    results = run_sequences(sequences=(1, 2, 3), seeds=(0, 1, 2),
                            backbone="resnet50", epochs=5, output=SAVE)

Tasks come from the saved seed-42 partition via `streams`. The method comes
from `hypernet`, `trainer` and `metrics`, and the Table 4 request sequences
from `config`.
"""

import argparse
import json
from pathlib import Path
import time

from .config import Config, dataset_defaults
from .experiment import run as run_requests
from .metrics import summary
from .streams import build_tasks
from .telemetry import RunLog, environment, learn_steps, progress_bar
from .tinyimagenet import DEFAULT_ROOT


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
                   on_request=None, verbose=True, progress=True,
                   checkpoint=True, resume=True,
                   config=None, **overrides) -> dict:
    """Work through a request sequence. Returns history, metrics and costs.

    The result is a dict with `config`, `history`, `numbers` (the paper's four),
    `environment` (device, versions, sizes, every Config field) and `costs`
    (per-request seconds and peak GPU memory, plus totals).

    `output` is a directory for the JSON files, or None to keep everything in
    memory. When it is set, the history and the cost log are rewritten after
    every request, so a long run can be watched and survives a crash. File
    names carry the sequence, backbone and seed, so a sweep over those does not
    overwrite itself; sweeping anything else still collides.

    `progress` shows tqdm bars when tqdm is installed: one counting the
    requests, and one inside each request counting optimizer steps, which is
    the only scale that moves while a long request runs. `verbose` prints a
    line per request and a summary at the end. `on_request` is called with each
    record on top of both. `config` takes a prepared Config and ignores the
    override arguments; `sequence` then only labels the output files.
    `download` fetches the dataset when it is missing, which a fresh Colab
    runtime needs.

    `checkpoint` writes the whole run state beside those files after every
    request, and `resume` picks it up if the run is started again. Together
    they mean a killed run continues from its last finished request rather
    than from the beginning. Both need `output`. The checkpoint is one file,
    overwritten in place, and it is as large as the hypernetwork: about 220 MB
    for ResNet50 at 200 chunks. Pass `checkpoint=False` to skip it, and
    `resume=False` to start over while keeping one.
    """
    if config is None:
        config = make_config(sequence, limit_requests, **overrides)
    elif overrides:
        raise ValueError("Pass overrides to make_config, or a finished config, not both")

    # Only the tasks these requests mention, because building one indexes both
    # splits. A full sequence names all twenty anyway.
    needed = sorted({task for _, task in config.requests}, key=int)
    setup_began = time.perf_counter()
    tasks = build_tasks(config, root=root, include=needed,
                        max_images=max_images, download=download)

    paths = {}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        stem = f"seq{sequence}_{config.backbone}_seed{config.seed}"
        paths = {
            "history": output / f"history_{stem}.json",
            "summary": output / f"summary_{stem}.json",
            "costs": output / f"costs_{stem}.json",
            "environment": output / f"environment_{stem}.json",
            "checkpoint": output / f"checkpoint_{stem}.pt",
        }

    log = RunLog(device=config.torch_device)
    facts = {}

    def record_environment(built):
        log.setup_seconds = time.perf_counter() - setup_began
        log.start()

        # A resumed run inherits the timings already recorded, so the totals
        # cover the whole run rather than only the part after the restart.
        if built.get("resumed_after"):
            # Without this the returned history, and the file written from it,
            # would hold only the requests since the restart.
            history.extend(built["history"])
            log.rows = built["costs"]
            log.setup_seconds += built["setup_seconds"]
            bar.update(built["resumed_after"])
            if verbose:
                print(f"resuming after {built['resumed_after']} completed "
                      f"requests, from {paths['checkpoint'].name}")

        facts.update(environment(config, built["hypernet"], built["tasks"]))
        if verbose:
            print(f"sequence {sequence}: {len(config.requests)} requests over "
                  f"{len(needed)} tasks, backbone {config.backbone}, "
                  f"{config.chunks} chunks, beta {config.beta}, gamma {config.gamma}")
            print(f"  {facts['hypernetwork_parameters']:,} hypernetwork parameters "
                  f"generating {facts['generated_parameters']:,}")
            print(f"  device {facts['device']}"
                  + (f", {facts['gpu']}, "
                     f"{facts['gpu_total_bytes'] / 2 ** 30:.1f} GiB" if facts["gpu"] else "")
                  + f", torch {facts['torch']}, commit {facts['git_commit']}")
        if paths:
            paths["environment"].write_text(
                json.dumps(facts, indent=2, default=str) + "\n", encoding="utf-8")

    history = []
    bar = progress_bar(len(config.requests), f"sequence {sequence}", progress)
    # Model construction happens inside run_requests, so close the setup clock
    # from the on_start hook rather than here.
    log.start()

    def record_done(record):
        history.append(record)
        steps = (log_steps(config, tasks, record["task"])
                 if record["action"] == "learn" else record["burn_in"])
        cost = log.measure(record, steps=steps)

        if verbose:
            # Through the bar, so the line does not land on top of it.
            bar.write(describe(record))
        bar.update(1)
        bar.set_postfix_str(
            f"{cost['seconds']:.0f}s"
            + (f", {cost['peak_memory_bytes'] / 2 ** 30:.1f} GiB"
               if cost["peak_memory_bytes"] else "")
        )
        if paths:
            paths["history"].write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8")
            log.write(paths["costs"])
        if on_request is not None:
            on_request(record)

    def collect_state(extra):
        """The telemetry the checkpoint should carry alongside the model."""
        extra["costs"] = log.rows
        extra["setup_seconds"] = log.setup_seconds

    try:
        run_requests(config, on_request=record_done, tasks=tasks,
                     on_start=record_environment, progress=progress,
                     checkpoint_path=paths.get("checkpoint") if checkpoint else None,
                     resume=resume, on_checkpoint=collect_state)
    finally:
        bar.close()

    numbers = summary(history)
    totals = log.totals()
    if paths:
        paths["summary"].write_text(
            json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
        log.write(paths["costs"])

    if verbose:
        chance = 100 / config.classes_per_task
        print()
        print(log.table())
        print()
        print(f"retain accuracy  {show(numbers['retain_accuracy'])}%")
        print(f"forget accuracy  {show(numbers['forget_accuracy'])}%  "
              f"(chance is {chance:.2f}%)")
        print(f"mean spill       {show(numbers['mean_spill'])}")
        print(f"mean relapse     {show(numbers['mean_relapse'])}")
        print()
        print(describe_costs(totals))
        if paths:
            print(f"\nSaved four JSON files to {output.resolve()}")

    return {
        "sequence": sequence,
        "config": config,
        "history": history,
        "numbers": numbers,
        "environment": facts,
        "costs": log.rows,
        "totals": totals,
    }


def log_steps(config, tasks, task) -> int:
    """Optimizer steps one learn request takes, for the per-step cost."""
    return learn_steps(config, len(tasks[task]["train"]))


def describe_costs(totals) -> str:
    """The cost summary as a few plain lines."""
    lines = [f"total time       {totals['total_seconds'] / 60:.1f} min, of which "
             f"{totals['setup_seconds']:.0f}s was setup",
             f"requests         {totals['request_seconds'] / 60:.1f} min "
             f"over {totals['requests']} requests"]
    for action in ("learn", "forget"):
        if action in totals:
            part = totals[action]
            lines.append(
                f"{action + ' requests':<16} {part['requests']:>2}  "
                f"{part['total_seconds'] / 60:>6.1f} min total, "
                f"{part['mean_seconds']:>6.1f} s mean, "
                f"{part['slowest_seconds']:>6.1f} s slowest")
    # The key is always there; its value is None when there was no GPU.
    if totals.get("peak_memory_bytes"):
        lines.append(f"peak GPU memory  {totals['peak_memory_bytes'] / 2 ** 30:.2f} GiB")
        lines.append(f"GPU time         {totals['gpu_hours']:.2f} hours")
    return "\n".join(lines)


def run_sequences(sequences=(1, 2, 3), seeds=(0,), output=None, verbose=True,
                  progress=True, **kwargs) -> list[dict]:
    """Run several sequences and seeds, and total up what they all cost.

    This is what the paper's tables are: sequence 1 over three seeds for the
    headline, all three sequences for the generalization check in Appendix H.
    Returns one result per run, and prints a comparison at the end.
    """
    runs = [(sequence, seed) for sequence in sequences for seed in seeds]
    results = []

    outer = progress_bar(len(runs), f"{len(runs)} runs", progress)
    for sequence, seed in runs:
        if verbose:
            print(f"\n{'=' * 64}\nsequence {sequence}, seed {seed}\n{'=' * 64}")
        results.append(run_experiment(
            sequence=sequence, seed=seed, output=output,
            verbose=verbose, progress=progress, **kwargs))
        outer.update(1)
    outer.close()

    if verbose:
        print()
        print(compare(results))

    if output is not None:
        path = Path(output) / "all_runs.json"
        path.write_text(json.dumps([
            {"sequence": r["sequence"], "seed": r["config"].seed,
             "backbone": r["config"].backbone,
             "numbers": r["numbers"], "totals": r["totals"]}
            for r in results
        ], indent=2) + "\n", encoding="utf-8")
        if verbose:
            print(f"\nSaved the combined log to {path.resolve()}")

    return results


def compare(results) -> str:
    """One row per run: the four numbers and what the run cost."""
    header = (f"{'seq':>3} {'seed':>4}  {'RA':>7} {'FA':>7} {'spill':>7} {'relapse':>8} "
              f"{'minutes':>8} {'peak GiB':>9}")
    lines = [header, "-" * len(header)]
    for result in results:
        numbers, totals = result["numbers"], result["totals"]
        peak = totals.get("peak_memory_bytes")
        lines.append(
            f"{result['sequence']:>3} {result['config'].seed:>4}  "
            f"{show(numbers['retain_accuracy']):>7} {show(numbers['forget_accuracy']):>7} "
            f"{show(numbers['mean_spill']):>7} {show(numbers['mean_relapse']):>8} "
            f"{totals['total_seconds'] / 60:>8.1f} "
            f"{('-' if not peak else f'{peak / 2 ** 30:.2f}'):>9}")

    total = sum(r["totals"]["total_seconds"] for r in results)
    lines.append("")
    lines.append(f"{len(results)} runs, {total / 3600:.2f} hours in total")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, nargs="+", default=[1], choices=(1, 2, 3),
                        help="which rows of the paper's Table 4; several are allowed")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
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
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "run")
    args = parser.parse_args()

    overrides = {
        name: value
        for name, value in (("beta", args.beta), ("device", args.device))
        if value is not None
    }
    run_sequences(
        sequences=tuple(args.sequence),
        seeds=tuple(args.seed),
        limit_requests=args.limit_requests,
        max_images=args.max_images,
        root=args.root,
        download=args.download,
        output=args.output,
        progress=not args.no_progress,
        backbone=args.backbone,
        epochs=args.epochs,
        chunks=args.chunks,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        **overrides,
    )


if __name__ == "__main__":
    main()
