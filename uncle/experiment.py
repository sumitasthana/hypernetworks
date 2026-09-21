"""Run a whole sequence of learn and forget requests."""

from collections.abc import Callable

import torch

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

    for index, (action, task) in enumerate(config.requests):
        before = {name: uncle.accuracy(name) for name in seen}

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

        record = {
            "index": index,
            "action": action,
            "task": task,
            "before": before,
            "after": {name: uncle.accuracy(name) for name in seen},
            "seen": list(seen),
            "forgotten": list(forgotten),
            "final_loss": losses[-1],
            "burn_in": burn_in,
        }
        history.append(record)

        if on_request is not None:
            on_request(record)

    return history
