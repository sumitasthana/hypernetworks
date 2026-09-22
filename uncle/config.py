"""Every knob in one place, with the paper's values as the defaults."""

from dataclasses import dataclass, field

import torch

# Appendix C, Table 4. The author runs three random request sequences per
# dataset. `L#n` means learn task n, `U#n` means unlearn task n.
PERMUTED_MNIST_SEQUENCES = {
    1: "L1 L0 U1 L5 L8 L9 L7 U0 L2 L3 L4 U8 U3 U5 L6",
    2: "L6 L7 L2 L1 L0 U1 L9 U7 U2 U0 L4 U4 L8 U6 L5",
    3: "L7 L1 L2 L8 L0 U1 L3 L6 U3 U2 L4 L5 U8 L9 U7",
}

FIVE_TASKS_SEQUENCES = {
    1: "L0 L1 U0 L2 L3 L4 U1",
    2: "L3 L4 L2 L0 L1 U3 U0",
    3: "L0 L2 U0 L4 L3 U2 U4",
}

# Tiny-ImageNet runs 30 requests over 20 tasks, twice as long as the rest.
TINY_IMAGENET_SEQUENCES = {
    1: "L3 L0 U3 L9 L5 L17 L1 L7 L14 L15 L19 U17 U7 L6 U15 U9 L12 L4 U5 U4 "
       "U6 U0 U1 U14 U12 L13 L18 L2 L11 L8",
    2: "L12 L13 L5 L8 L2 U8 L14 U13 U5 U2 L3 U3 L16 U12 L11 U16 L7 L15 L10 L19 "
       "L9 U14 U7 L18 L6 L1 L0 L4 U6 L17",
    3: "L2 L7 U2 L18 L12 U7 U18 L16 L0 U16 U0 L13 L4 U12 U13 L9 L19 U19 U4 L10 "
       "L14 L5 U5 U10 L11 L1 U1 L17 L6 L3",
}

# Per-dataset defaults from the paper: the request sequences, how many tasks
# the sequences address, beta (Appendix C: 1e-1 for Permuted MNIST and
# CIFAR-100, 1e-2 for Tiny-ImageNet, 1e-3 for 5-Tasks), and the backbone
# ("ResNet18 in the case of Permuted MNIST experiments and ResNet50 elsewhere
# to demonstrate scalability"). The backbone belongs here rather than in the
# dataclass default: leaving it to the global default silently gave Tiny
# ImageNet a ResNet18, which is not the paper's setting and says nothing when
# it happens.


DATASETS = {
    "permuted_mnist": {
        "channels": 1, "size": 28, "task_count": 10, "beta": 0.1,
        "backbone": "resnet18", "sequences": PERMUTED_MNIST_SEQUENCES,
    },
    "tiny_imagenet": {
        "channels": 3, "size": 64, "task_count": 20, "beta": 0.01,
        "backbone": "resnet50", "sequences": TINY_IMAGENET_SEQUENCES,
    },
}


def parse_sequence(text: str) -> tuple[tuple[str, str], ...]:
    """Turn "L1 L0 U1" into (("learn", "1"), ("learn", "0"), ("forget", "1"))."""
    actions = {"L": "learn", "U": "forget"}
    return tuple(
        (actions[token[0]], token[1:]) for token in text.split()
    )


def dataset_defaults(dataset: str, sequence: int) -> dict:
    """The paper's task list, request sequence, beta and backbone for a dataset.

    Sequence numbers are the rows of Table 4. The task list is sized to the
    sequence: Permuted MNIST names tasks 0-9, Tiny-ImageNet names 0-19.
    Everything here is a value the paper states, so a caller that overrides one
    is departing from the paper on purpose.
    """
    settings = DATASETS[dataset]
    return {
        "dataset": dataset,
        "tasks": tuple(str(index) for index in range(settings["task_count"])),
        "requests": parse_sequence(settings["sequences"][sequence]),
        "beta": settings["beta"],
        "backbone": settings["backbone"],
    }


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Config:
    """Settings for one continual learn and unlearn experiment.

    The defaults are the paper's Permuted-MNIST setting: ResNet18 generated in
    200 chunks, 10 tasks, and request sequence 1 from Table 4. That needs a
    GPU. For a quick CPU run pass backbone="cnn" with fewer chunks and tasks.
    """

    dataset: str = "permuted_mnist"   # see DATASETS
    tasks: tuple[str, ...] = tuple(str(index) for index in range(10))
    requests: tuple[tuple[str, str], ...] = parse_sequence(
        PERMUTED_MNIST_SEQUENCES[1]
    )

    # Tiny ImageNet only. The paper gives 10 tasks of 10 classes but never says
    # which classes. The grouping lives in uncle/task_partition.json: the 200
    # WordNet IDs shuffled with seed 42 into 20 disjoint groups of ten, written
    # once and verified on every run.
    classes_per_task: int = 10

    backbone: str = "resnet18"        # "resnet18", "resnet50", or "cnn"
    seed: int = 0
    epochs: int = 5
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 1e-3       # Adam, the paper's value

    beta: float = 0.1     # hold-other-tasks-still term, eq. 2. Paper: 0.1 for PMNIST
    gamma: float = 0.01   # push-toward-noise term, eq. 3. Paper: 0.01 for PMNIST

    code_dim: int = 32               # Appendix B: task and chunk codes are both 32
    chunks: int = 200                # Appendix B: 200 chunks per task network
    hidden: tuple[int, ...] = (128, 256, 512)
    noise_samples: int = 10          # n in eq. 3

    # Appendix C: burn-in starts at 100 and drops 10% after each unlearn,
    # never below 20.
    burn_in: int = 100
    burn_in_decay: float = 0.9
    burn_in_min: int = 20

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

        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        if self.classes_per_task < 2:
            raise ValueError("classes_per_task must be at least 2.")
        if self.backbone not in ("resnet18", "resnet50", "cnn"):
            raise ValueError(f"Unknown backbone: {self.backbone}")
        if self.chunks < 1:
            raise ValueError("chunks must be at least 1.")
        if not self.hidden:
            raise ValueError("hidden must name at least one layer width.")
        if not 0 < self.burn_in_decay <= 1:
            raise ValueError("burn_in_decay must be in (0, 1].")

    @property
    def input_channels(self) -> int:
        return DATASETS[self.dataset]["channels"]

    @property
    def input_size(self) -> int:
        return DATASETS[self.dataset]["size"]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def burn_in_for(self, completed_forgets: int) -> int:
        """Appendix C's annealed burn-in for the next unlearn request."""
        annealed = int(self.burn_in * self.burn_in_decay ** completed_forgets)
        return max(self.burn_in_min, annealed)
