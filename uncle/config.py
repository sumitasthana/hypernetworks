"""Every knob in one place."""

from dataclasses import dataclass, field

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Config:
    """Settings for one continual learn and unlearn experiment.

    A request is a pair such as ("learn", "A") or ("forget", "A").
    """

    tasks: tuple[str, ...] = ("A", "B", "C")
    requests: tuple[tuple[str, str], ...] = (
        ("learn", "A"),
        ("learn", "B"),
        ("forget", "A"),
        ("learn", "C"),
    )

    seed: int = 0
    epochs: int = 1
    batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 1e-3

    beta: float = 0.1     # strength of the hold-other-tasks-still term (paper eq. 2)
    gamma: float = 0.01   # strength of the push-toward-noise term (paper eq. 3)

    code_dim: int = 32               # length of a task code and of a chunk code
    chunks: int = 32                 # how many slices the target weights arrive in
    hidden: tuple[int, ...] = (128, 256, 512)
    noise_samples: int = 10
    burn_in: int = 100               # optimizer steps spent on one forget request

    data_root: str = "./data"
    device: str = field(default_factory=_default_device)

    def __post_init__(self) -> None:
        # The request list is the part a user edits most, so check it properly.
        learned: set[str] = set()
        forgotten: set[str] = set()

        for action, task in self.requests:
            if action not in ("learn", "forget"):
                raise ValueError(f"Unknown action: {action}")
            if task not in self.tasks:
                raise ValueError(f"Request names an unknown task: {task}")

            if action == "learn":
                if task in learned:
                    raise ValueError(f"Task {task} is learned twice.")
                learned.add(task)
            else:
                if task not in learned:
                    raise ValueError(f"Task {task} is forgotten before it is learned.")
                if task in forgotten:
                    raise ValueError(f"Task {task} is forgotten twice.")
                forgotten.add(task)

        if self.chunks < 1:
            raise ValueError("chunks must be at least 1.")
        if not self.hidden:
            raise ValueError("hidden must name at least one layer width.")

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)
