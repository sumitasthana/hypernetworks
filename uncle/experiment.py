"""Run a whole sequence of learn and forget requests."""

from collections.abc import Callable

import torch

from .config import Config
from .data import build_tasks
from .hypernet import HyperNetwork, build_target
from .trainer import UnCLe


def run(config: Config, on_request: Callable[[dict], None] | None = None) -> list[dict]:
    """Work through config.requests and return one record per request.

    `on_request` is called with each record as it completes, so a caller can
    print or log progress without this function knowing how.
    """
    torch.manual_seed(config.seed)

    tasks = build_tasks(config)
    target = build_target(config.torch_device)
    hypernet = HyperNetwork(target, config)
    uncle = UnCLe(hypernet, config, target, tasks)

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
            seen.append(task)
        else:
            losses = uncle.forget(task, protected)
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
        }
        history.append(record)

        if on_request is not None:
            on_request(record)

    return history
