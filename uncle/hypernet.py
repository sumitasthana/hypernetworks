"""The network that produces another network's weights.

Follows the architecture in the paper's Appendix B: parameters are generated
in chunks, a chunk code is concatenated with the task code to form one
task-chunk pair per chunk, and the last layer is split into one head per
parameter type.
"""

import copy
import math

import torch
from torch import nn
from torchvision.models import resnet18, resnet50

from .config import Config

BATCHNORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def build_target(config: Config) -> nn.Module:
    """The network whose weights get generated: F in the paper.

    It is never trained. Every forward pass receives a fresh set of weights, so
    its own values are only a template for the shapes.

    The paper uses ResNet18 for Permuted-MNIST and ResNet50 elsewhere. The
    "cnn" option is a small stand-in for quick CPU runs; it has no BatchNorm,
    so it also has no per-task running statistics to carry around.
    """
    if config.backbone == "cnn":
        target = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
        )
    else:
        builder = {"resnet18": resnet18, "resnet50": resnet50}[config.backbone]
        target = builder(weights=None, num_classes=10)

        # Permuted-MNIST is one channel at 28x28. The stock 7x7 stride-2 stem
        # with a max-pool would throw most of that away.
        target.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        target.maxpool = nn.Identity()

    return target.to(config.torch_device).requires_grad_(False)


def parameter_group(module: nn.Module, module_name: str) -> str:
    """Which output head generates this parameter.

    Appendix B splits the hypernetwork's final layer into heads by parameter
    type: batch normalization parameters, residual connection parameters, and
    ordinary network weights.
    """
    if isinstance(module, BATCHNORM):
        return "batchnorm"
    if "downsample" in module_name or "shortcut" in module_name:
        return "residual"
    return "weights"


def allocate_chunks(sizes: dict[str, int], chunks: int) -> dict[str, int]:
    """Share the chunk budget across the heads.

    The paper fixes the total at 200 chunks per task-specific network but does
    not say how they are split between heads. This gives every head at least
    one chunk and divides the rest in proportion to how many numbers it owns.
    """
    if chunks < len(sizes):
        raise ValueError(f"chunks must be at least {len(sizes)}, one per head.")

    total = sum(sizes.values())
    spare = chunks - len(sizes)
    shares = {group: spare * size / total for group, size in sizes.items()}
    counts = {group: 1 + math.floor(share) for group, share in shares.items()}

    # Hand the rounding leftovers to the heads with the largest fractions.
    leftover = chunks - sum(counts.values())
    ranked = sorted(shares, key=lambda group: shares[group] % 1, reverse=True)
    for group in ranked[:leftover]:
        counts[group] += 1

    return counts


class HyperNetwork(nn.Module):
    """Turns a short task code into a full set of weights for `target`.

    This is H(.; phi) in the paper. It holds four trainable things: the shared
    trunk, one output head per parameter type, one code per chunk, and one code
    per task.
    """

    def __init__(self, target: nn.Module, config: Config):
        super().__init__()
        self.config = config

        modules = dict(target.named_modules())

        # Where each weight tensor sits inside its head's output, and how much
        # to shrink it. The shrink factors turn unit-spread raw numbers into a
        # textbook He initialization, so the generated network starts out
        # correctly scaled.
        self.layout: dict[str, tuple[str, int, torch.Size]] = {}
        self.scales: dict[str, float] = {}
        sizes: dict[str, int] = {}

        for name, parameter in target.named_parameters():
            module_name = name.rpartition(".")[0]
            group = parameter_group(modules[module_name], module_name)

            start = sizes.get(group, 0)
            self.layout[name] = (group, start, parameter.shape)
            sizes[group] = start + parameter.numel()

            fan_in = parameter[0].numel() if parameter.dim() > 1 else 1
            self.scales[name] = (2 / fan_in) ** 0.5 if parameter.dim() > 1 else 0.01

        self.group_sizes = sizes
        self.group_order = list(sizes)
        self.total = sum(sizes.values())

        self.chunk_counts = allocate_chunks(sizes, config.chunks)
        self.chunk_widths = {
            group: -(-sizes[group] // self.chunk_counts[group]) for group in sizes
        }

        # Which rows of chunk_codes belong to which head.
        self.chunk_slices: dict[str, slice] = {}
        first = 0
        for group in self.group_order:
            count = self.chunk_counts[group]
            self.chunk_slices[group] = slice(first, first + count)
            first += count

        widths = [2 * config.code_dim, *config.hidden]
        layers: list[nn.Module] = []
        for inputs, outputs in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(inputs, outputs), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        self.heads = nn.ModuleDict({
            group: nn.Linear(widths[-1], self.chunk_widths[group])
            for group in self.group_order
        })

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        # Heads aim for raw outputs with spread near 1, so that after the
        # scale factors the generated network starts out correctly initialized.
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=widths[-1] ** -0.5)
            nn.init.zeros_(head.bias)

        self.chunk_codes = nn.Parameter(
            torch.randn(config.chunks, config.code_dim, device=config.torch_device)
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

    def _raw_groups(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each head's unscaled output, padding removed."""
        generated = {}

        for group in self.group_order:
            codes = self.chunk_codes[self.chunk_slices[group]]
            # One task-chunk pair per chunk, all run through the trunk at once.
            pairs = torch.cat([code.expand(codes.shape[0], -1), codes], dim=1)
            raw = self.heads[group](self.trunk(pairs)).reshape(-1)
            generated[group] = raw[:self.group_sizes[group]]

        return generated

    def raw_from_code(self, code: torch.Tensor) -> torch.Tensor:
        """Every generated number before scaling, as one flat vector."""
        generated = self._raw_groups(code)
        return torch.cat([generated[group] for group in self.group_order])

    def weights_from_code(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        """One task code in, a full set of target weights out: theta = H(e; phi)."""
        generated = self._raw_groups(code)
        return {
            name: generated[group][start:start + shape.numel()].reshape(shape)
                  * self.scales[name]
            for name, (group, start, shape) in self.layout.items()
        }

    def weights_for(self, task: str) -> dict[str, torch.Tensor]:
        return self.weights_from_code(self.task_codes[task])

    def raw_for(self, task: str) -> torch.Tensor:
        return self.raw_from_code(self.task_codes[task])

    def generator_parameters(self) -> list:
        """The weights that learning and forgetting update: phi in the paper.

        Task codes and chunk codes are handled separately, so this is its own
        method rather than `self.parameters()`. Anything presenting itself to
        `UnCLe` as a hypernetwork must offer this.
        """
        return [*self.trunk.parameters(), *self.heads.parameters()]

    # -- snapshots -----------------------------------------------------------

    def snapshot(self) -> "HyperNetwork":
        """A frozen copy of this network as it is right now: phi* in the paper.

        Learning and forgetting both compare against a snapshot taken before
        the request started. That comparison is the paper's regularizer.
        """
        return copy.deepcopy(self).eval().requires_grad_(False)
