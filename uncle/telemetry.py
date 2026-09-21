"""What a run cost: time, GPU memory, and the sizes that explain both.

A learn request and a forget request are very different jobs. Learning walks
the whole dataset for several epochs; forgetting takes a fixed number of steps
and touches no data at all. Averaging them together hides the thing worth
knowing, so everything here is recorded per request and summarized per action.

Peak memory is the number to watch on a long sequence. The regularizer
regenerates every protected task's weights inside one graph, so the cost of a
request grows with how many tasks came before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import platform
import subprocess
import time

import torch


def progress_bar(total, description, enabled=True, leave=True):
    """A tqdm bar where tqdm is installed, and a do-nothing stand-in where not.

    tqdm ships with Colab and is not worth making a hard requirement for a
    progress bar, so a missing import is not an error. The stand-in carries
    `write` as well, so callers can print through the bar without having to
    know which one they hold.
    """
    if enabled:
        try:
            from tqdm.auto import tqdm
            return tqdm(total=total, desc=description, unit="step", leave=leave)
        except ImportError:
            pass

    class Quiet:
        def update(self, n=1): pass
        def set_postfix_str(self, text): pass
        def write(self, text): print(text, flush=True)
        def close(self): pass

    return Quiet()


def gpu_name() -> str | None:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def gpu_total_memory() -> int | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory


def git_commit() -> str | None:
    """The commit this ran on, so a number can be traced back to code."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment(config, hypernet=None, tasks=None) -> dict:
    """Everything needed to make sense of the numbers a run produces later."""
    report = {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(config.torch_device),
        "gpu": gpu_name(),
        "gpu_total_bytes": gpu_total_memory(),
        "config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in vars(config).items()
        },
    }
    if hypernet is not None:
        report["hypernetwork_parameters"] = sum(p.numel() for p in hypernet.parameters())
        report["generated_parameters"] = hypernet.total
        report["chunks_per_head"] = dict(hypernet.chunk_counts)
    if tasks is not None:
        report["task_sizes"] = {
            name: {split: len(dataset) for split, dataset in splits.items()}
            for name, splits in tasks.items()
        }
    return report


@dataclass
class RunLog:
    """Times each request and records what the GPU did during it.

    Call `start()` once, then `measure(record)` as each request completes. The
    clock runs from the end of the previous request, which is what you want:
    the gap covers the request plus the accuracy sweep that follows it, and
    that sweep is part of what a run costs.
    """

    device: torch.device
    rows: list[dict] = field(default_factory=list)
    _mark: float = 0.0
    _began: float = 0.0

    @property
    def on_gpu(self) -> bool:
        return self.device.type == "cuda"

    def start(self) -> None:
        if self.on_gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._began = self._mark = time.perf_counter()

    def measure(self, record: dict, steps: int | None = None) -> dict:
        """Close the books on one request and open them for the next."""
        if self.on_gpu:
            # Kernels are queued, so without this the clock stops too early.
            torch.cuda.synchronize()

        now = time.perf_counter()
        row = {
            "index": record["index"],
            "action": record["action"],
            "task": record["task"],
            # Always one fewer than the tasks seen: a request never protects
            # its own task. `seen` already includes it, for a learn because it
            # was just added and for a forget because it is still on the list.
            "protected": max(len(record["seen"]) - 1, 0),
            "steps": steps,
            "seconds": now - self._mark,
            "elapsed_seconds": now - self._began,
            "peak_memory_bytes": (
                torch.cuda.max_memory_allocated() if self.on_gpu else None
            ),
        }
        if steps:
            row["seconds_per_step"] = row["seconds"] / steps
        self.rows.append(row)

        self._mark = now
        if self.on_gpu:
            torch.cuda.reset_peak_memory_stats()
        return row

    # -- reading it back -----------------------------------------------------

    def totals(self) -> dict:
        """Per-action totals, plus the worst request, which is the binding one."""
        report = {
            "total_seconds": sum(row["seconds"] for row in self.rows),
            "requests": len(self.rows),
        }
        for action in ("learn", "forget"):
            rows = [row for row in self.rows if row["action"] == action]
            if not rows:
                continue
            seconds = [row["seconds"] for row in rows]
            report[action] = {
                "requests": len(rows),
                "total_seconds": sum(seconds),
                "mean_seconds": sum(seconds) / len(rows),
                "slowest_seconds": max(seconds),
            }
        if self.on_gpu and self.rows:
            peaks = [row["peak_memory_bytes"] for row in self.rows]
            report["peak_memory_bytes"] = max(peaks)
            report["gpu_hours"] = report["total_seconds"] / 3600
        return report

    def table(self) -> str:
        """The per-request log as fixed-width text, for a notebook or a log file."""
        header = (f"{'#':>3}  {'action':<6} {'task':>4} {'prot':>4} {'steps':>6} "
                  f"{'seconds':>9} {'s/step':>7} {'peak GiB':>9}")
        lines = [header, "-" * len(header)]

        for row in self.rows:
            peak = row["peak_memory_bytes"]
            steps = "-" if row["steps"] is None else str(row["steps"])
            per_step = row.get("seconds_per_step")
            lines.append(
                f"{row['index']:>3}  {row['action']:<6} {row['task']:>4} "
                f"{row['protected']:>4} {steps:>6} {row['seconds']:>9.1f} "
                f"{('-' if per_step is None else f'{per_step:.3f}'):>7} "
                f"{('-' if not peak else f'{peak / 2 ** 30:.2f}'):>9}"
            )
        return "\n".join(lines)

    def write(self, path) -> None:
        path.write_text(
            json.dumps({"requests": self.rows, "totals": self.totals()}, indent=2) + "\n",
            encoding="utf-8",
        )


def learn_steps(config, task_size: int) -> int:
    """Optimizer steps one learn request will take, for the per-step cost."""
    batches = -(-task_size // config.batch_size)
    return batches * config.epochs
