"""Run a whole sequence of learn and forget requests."""

from collections.abc import Callable

import torch

from . import checkpoint as checkpointing
from .config import Config
from .data import build_tasks
from .hypernet import HyperNetwork, build_target
from .trainer import UnCLe


def run(
    config: Config,
    on_request: Callable[[dict], None] | None = None,
    tasks: dict | None = None,
    on_start: Callable[[dict], None] | None = None,
    progress: bool = False,
    checkpoint_path=None,
    resume: bool = True,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Work through config.requests and return one record per request.

    `on_request` is called with each record as it completes, so a caller can
    print or log progress without this function knowing how.

    `tasks` lets a caller supply its own {task: {"train", "test"}} datasets
    instead of the ones this config would build.

    `progress` puts a bar on each request's optimizer steps, which is the only
    scale that says anything while a forty-minute request is running.

    `on_start` is called once with the built target, hypernetwork and tasks,
    before the first request. It exists so a caller can record what it is about
    to run, sizes included, without building any of it a second time.

    `checkpoint_path` saves the run after every request and, with `resume`,
    picks up an existing checkpoint instead of starting over. Resuming skips
    the requests already in the checkpoint's history. `on_checkpoint` is called
    with the state about to be written, so a caller can add its own fields.
    """
    torch.manual_seed(config.seed)

    tasks = build_tasks(config) if tasks is None else tasks
    target = build_target(config)
    hypernet = HyperNetwork(target, config)
    uncle = UnCLe(hypernet, config, target, tasks, progress=progress)

    if on_start is not None:
        on_start({"config": config, "target": target,
                  "hypernet": hypernet, "tasks": tasks, "uncle": uncle})

    history: list[dict] = []
    seen: list[str] = []
    forgotten: list[str] = []
    previous: dict[str, float] = {}
    done = 0

    saved = (checkpointing.load(checkpoint_path, config)
             if checkpoint_path is not None and resume else None)
    if saved is not None:
        done = checkpointing.restore(saved, hypernet=hypernet, uncle=uncle)
        history = saved["history"]
        seen, forgotten = saved["seen"], saved["forgotten"]
        previous = saved["previous"]
        if on_start is not None:
            on_start({"config": config, "target": target, "hypernet": hypernet,
                      "tasks": tasks, "uncle": uncle, "resumed_after": done,
                      "history": list(history), "costs": saved["costs"],
                      "setup_seconds": saved["setup_seconds"]})

    for index, (action, task) in enumerate(config.requests):
        if index < done:
            continue        # already in the checkpoint's history
        # Nothing touches the model between requests, so the accuracies taken
        # after the last one are still current. Measuring them again doubled
        # the evaluation work for an identical answer.
        before = previous if previous else {name: uncle.accuracy(name) for name in seen}

        # Every task met so far, except the one this request is about.
        # Forgotten tasks stay on the list, which is what stops them relapsing.
        protected = [name for name in seen if name != task]

        if action == "learn":
            losses = uncle.learn(task, protected)
            burn_in = None
            seen.append(task)
        else:
            # Appendix C anneals the burn-in by 10% after each unlearn.
            burn_in = config.burn_in_for(len(forgotten))
            losses = uncle.forget(task, protected, burn_in=burn_in)
            forgotten.append(task)

        after = {name: uncle.accuracy(name) for name in seen}
        previous = dict(after)

        record = {
            "index": index,
            "action": action,
            "task": task,
            "before": before,
            "after": after,
            "seen": list(seen),
            "forgotten": list(forgotten),
            "final_loss": losses[-1],
            "burn_in": burn_in,
        }
        history.append(record)

        if on_request is not None:
            on_request(record)

        # After the callback, so the checkpoint carries whatever telemetry the
        # callback recorded, but inside the loop so the request is never lost.
        if checkpoint_path is not None:
            extra = {"costs": [], "setup_seconds": 0.0}
            if on_checkpoint is not None:
                on_checkpoint(extra)
            checkpointing.save(
                checkpoint_path, config=config, hypernet=hypernet, uncle=uncle,
                history=history, seen=seen, forgotten=forgotten,
                previous=previous, costs=extra["costs"],
                setup_seconds=extra["setup_seconds"],
            )

    return history
