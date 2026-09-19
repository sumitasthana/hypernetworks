"""The network that produces another network's weights."""

import copy

import torch
from torch import nn

from .config import Config


def build_target(device: torch.device) -> nn.Module:
    """The CNN whose weights get generated.

    It is never trained. Every forward pass receives a fresh set of weights, so
    its own values are only a template for the shapes. No normalization layers,
    which is what keeps this file free of per-task running statistics.
    """
    target = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
    )
    return target.to(device).requires_grad_(False)


class HyperNetwork(nn.Module):
    """Turns a short task code into a full set of weights for `target`.

    Holds three trainable things: the shared trunk, one code per chunk of the
    target's weights, and one code per task.
    """

    def __init__(self, target: nn.Module, config: Config):
        super().__init__()
        self.config = config
        self.chunks = config.chunks

        # Where each weight tensor sits inside one long flat vector, and how
        # much to shrink it. The shrink factors turn unit-spread raw numbers
        # into a textbook He initialization, which is what an untrained
        # network looks like. That is the state a forget request returns to.
        self.layout: dict[str, tuple[int, torch.Size]] = {}
        self.scales: dict[str, float] = {}
        offset = 0
        for name, parameter in target.named_parameters():
            self.layout[name] = (offset, parameter.shape)
            fan_in = parameter[0].numel() if parameter.dim() > 1 else 1
            self.scales[name] = (2 / fan_in) ** 0.5 if parameter.dim() > 1 else 0.01
            offset += parameter.numel()

        self.total = offset
        self.width = -(-self.total // self.chunks)  # per chunk, rounded up

        widths = [2 * config.code_dim, *config.hidden]
        layers: list[nn.Module] = []
        for inputs, outputs in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(inputs, outputs), nn.ReLU()]
        layers.append(nn.Linear(widths[-1], self.width))
        self.trunk = nn.Sequential(*layers)

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        # The last layer aims for raw outputs with spread near 1, so that after
        # the shrink factors the generated CNN starts out correctly initialized.
        nn.init.normal_(self.trunk[-1].weight, std=widths[-1] ** -0.5)

        self.chunk_codes = nn.Parameter(
            torch.randn(self.chunks, config.code_dim, device=config.torch_device)
        )
        self.task_codes = nn.ParameterDict()
        self.to(config.torch_device)

    # -- codes ---------------------------------------------------------------

    def add_task(self, task: str) -> nn.Parameter:
        """Create a fresh random code for a task that has not been seen."""
        if task in self.task_codes:
            raise ValueError(f"Task {task} already has a code.")

        self.task_codes[task] = nn.Parameter(
            torch.randn(self.config.code_dim, device=self.config.torch_device)
        )
        return self.task_codes[task]

    @property
    def known_tasks(self) -> list[str]:
        return list(self.task_codes)

    # -- generation ----------------------------------------------------------

    def raw_from_code(self, code: torch.Tensor) -> torch.Tensor:
        """One task code in, one flat vector of unscaled numbers out."""
        # Pair the task code with every chunk code, then run all chunks at once.
        pairs = torch.cat([code.expand(self.chunks, -1), self.chunk_codes], dim=1)
        return self.trunk(pairs).reshape(-1)[:self.total]

    def weights_from_code(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        """One task code in, a full set of target weights out."""
        raw = self.raw_from_code(code)
        return {
            name: raw[start:start + shape.numel()].reshape(shape) * self.scales[name]
            for name, (start, shape) in self.layout.items()
        }

    def weights_for(self, task: str) -> dict[str, torch.Tensor]:
        return self.weights_from_code(self.task_codes[task])

    def raw_for(self, task: str) -> torch.Tensor:
        return self.raw_from_code(self.task_codes[task])

    # -- snapshots -----------------------------------------------------------

    def snapshot(self) -> "HyperNetwork":
        """A frozen copy of this network as it is right now.

        Learning and forgetting both compare against a snapshot taken before
        the request started. That comparison is the paper's regularizer.
        """
        return copy.deepcopy(self).eval().requires_grad_(False)
