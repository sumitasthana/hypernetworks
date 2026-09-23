"""Save a run mid-sequence, and pick it up where it stopped.

A thirty-request ResNet50 sequence is long enough that something will end it
early: a disconnected runtime, a session limit, a full disk. Without this the
only choice is to start again from request 0.

What has to be saved is more than the hypernetwork. The per-task BatchNorm
statistics live outside it, in the trainer, because the hypernetwork generates
parameters and not buffers. The request loop's own bookkeeping has to travel
too, or a resumed run would not know which tasks it had seen, which it had
forgotten, or what the accuracies were before the next request.

The random number generator state goes in as well, so a resumed run makes the
same draws an uninterrupted one would have made. That is what lets a resumed
run produce the same records rather than merely similar ones.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

#: Bumped when the saved shape changes, so an old file is refused rather than
#: half-loaded into a mismatched run.
FORMAT = 1


def save(path, *, config, hypernet, uncle, history, seen, forgotten,
         previous, costs, setup_seconds) -> None:
    """Write one checkpoint, replacing any earlier one.

    Written to a temporary name and moved into place, because the alternative
    is a half-written file exactly when the runtime dies, which is the moment
    this exists for.
    """
    path = Path(path)
    payload = {
        "format": FORMAT,
        # Every field, not a chosen few. A sweep over beta or gamma keeps the
        # sequence, backbone and seed, so it writes to the same file name;
        # comparing only those would resume one sweep point inside the next
        # and produce numbers belonging to neither.
        "config": dict(vars(config)),
        "tasks_with_codes": list(hypernet.task_codes),
        "hypernet": hypernet.state_dict(),
        "task_buffers": uncle.task_buffers,
        "history": history,
        "seen": list(seen),
        "forgotten": list(forgotten),
        "previous": dict(previous),
        "costs": costs,
        "setup_seconds": setup_seconds,
        "rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
    }
    temporary = path.with_suffix(path.suffix + ".writing")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path, config=None):
    """Read a checkpoint, or return None when there is nothing to resume.

    Refuses a checkpoint whose config differs anywhere at all. Resuming one
    run inside another would produce numbers belonging to neither, and a
    silent mismatch is worse than starting over. This matters most for a
    sweep: changing beta or gamma keeps the file name, because only the
    sequence, backbone and seed are in it.

    With config=None, read the saved configuration without a resume comparison.
    Diagnostic callers use that configuration to build the saved architecture.
    Only load trusted checkpoints, which can contain Python objects.
    """
    path = Path(path)
    if not path.exists():
        return None

    # Keep RNG states on CPU. Model tensors are copied to the destination by
    # load_state_dict; buffers are moved explicitly in restore.
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if payload.get("format") != FORMAT:
        raise ValueError(
            f"{path} was written in format {payload.get('format')}, "
            f"this code reads {FORMAT}. Delete it to start over.")

    # Older checkpoints predate the optional forgetting-only learning rate.
    # None preserves their original behavior and is the only migrated default.
    payload["config"].setdefault("forgetting_learning_rate", None)
    if config is None:
        return payload

    now = dict(vars(config))
    saved = payload["config"]
    mismatch = sorted(
        key for key in set(saved) | set(now) if saved.get(key) != now.get(key)
    )
    if mismatch:
        raise ValueError(
            f"{path} is a checkpoint of a different run: {', '.join(mismatch)} "
            f"differ. Give this run its own output directory, pass "
            f"output=None, or delete that file.")

    return payload


def restore(payload, *, hypernet, uncle) -> int:
    """Put a checkpoint back into a freshly built run. Returns requests done.

    Task codes have to exist before the state dict can fill them, since they
    are a ParameterDict that starts empty and grows one entry per learned
    task.
    """
    cpu_rng = _rng_state(payload["rng"], "CPU")
    cuda_rng = payload["cuda_rng"]
    if cuda_rng is not None and torch.cuda.is_available():
        if len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count differs from this runtime.")
        # Never skip an entry: its position identifies the corresponding GPU.
        cuda_rng = [_rng_state(state, f"CUDA device {i}")
                    for i, state in enumerate(cuda_rng)]

    for task in payload["tasks_with_codes"]:
        if task not in hypernet.task_codes:
            hypernet.add_task(task)

    hypernet.load_state_dict(payload["hypernet"])
    uncle.task_buffers = {
        task: {name: value.to(uncle.device).clone() for name, value in buffers.items()}
        for task, buffers in payload["task_buffers"].items()
    }
    uncle._reference = {}

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)

    return len(payload["history"])


def _rng_state(state, name):
    """Normalize device placement without silently converting corrupt state."""
    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError(f"{name} RNG state must be a one-dimensional uint8 tensor.")
    return state.detach().cpu().contiguous()
