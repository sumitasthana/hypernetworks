"""Checks that run in seconds and never touch MNIST.

    python tests/test_uncle.py

Uses a tiny synthetic dataset: each class is a bright square in its own spot.
Trivially learnable, which is the point. These check the machinery, not accuracy.
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uncle import Config, HyperNetwork, UnCLe, build_target  # noqa: E402
from uncle.metrics import relapse, spill, summary  # noqa: E402


class Blocks(Dataset):
    """Class k is a bright 5x5 square at grid position k."""

    def __init__(self, count: int, shift: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 10, (count,), generator=generator)
        self.images = torch.zeros(count, 1, 28, 28)

        for index, label in enumerate(self.labels):
            row = (int(label) // 5) * 10 + shift
            column = (int(label) % 5) * 5
            self.images[index, 0, row:row + 5, column:column + 5] = 1.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], int(self.labels[index])


def build(**overrides):
    config = Config(
        tasks=("A", "B"),
        requests=(("learn", "A"), ("learn", "B"), ("forget", "A")),
        backbone="cnn", chunks=32,
        epochs=25, batch_size=64, burn_in=150, device="cpu",
        **overrides,
    )
    torch.manual_seed(config.seed)

    tasks = {
        "A": {"train": Blocks(256, 0, 1), "test": Blocks(128, 0, 2)},
        "B": {"train": Blocks(256, 3, 3), "test": Blocks(128, 3, 4)},
    }
    target = build_target(config)
    hypernet = HyperNetwork(target, config)

    return UnCLe(hypernet, config, target, tasks), hypernet, target


def test_config_rejects_bad_requests():
    for bad in (
        (("forget", "A"),),                    # forgotten before learned
        (("learn", "A"), ("learn", "A")),      # learned twice
        (("learn", "Z"),),                     # unknown task
        (("rename", "A"),),                    # unknown action
    ):
        try:
            Config(tasks=("A", "B"), requests=bad)
        except ValueError:
            continue
        raise AssertionError(f"Config accepted a bad request list: {bad}")


def test_generated_weights_fit_the_target():
    _, hypernet, target = build()
    hypernet.add_task("A")
    weights = hypernet.weights_for("A")

    expected = dict(target.named_parameters())
    assert set(weights) == set(expected), "generated names do not match the target"

    for name, value in weights.items():
        assert value.shape == expected[name].shape, f"{name} has the wrong shape"
        assert torch.isfinite(value).all(), f"{name} is not finite"


def test_heads_split_by_parameter_type():
    """Appendix B: one head for weights, one for BatchNorm, one for residuals."""
    config = Config(
        tasks=("A",), requests=(("learn", "A"),),
        backbone="cnn", chunks=16, device="cpu",
    )

    resnet_shaped = nn.Sequential()
    resnet_shaped.add_module("conv", nn.Conv2d(1, 4, 3, padding=1, bias=False))
    resnet_shaped.add_module("bn", nn.BatchNorm2d(4))
    resnet_shaped.add_module("downsample", nn.Conv2d(4, 8, 1, bias=False))
    resnet_shaped.add_module("head", nn.Linear(8 * 28 * 28, 10))

    hypernet = HyperNetwork(resnet_shaped, config)

    assert set(hypernet.heads) == {"weights", "batchnorm", "residual"}
    assert hypernet.layout["bn.weight"][0] == "batchnorm"
    assert hypernet.layout["downsample.weight"][0] == "residual"
    assert hypernet.layout["conv.weight"][0] == "weights"

    # Every head gets at least one chunk and the budget is spent exactly.
    assert sum(hypernet.chunk_counts.values()) == config.chunks
    assert all(count >= 1 for count in hypernet.chunk_counts.values())

    # Generation still reproduces every tensor of the target, exactly once.
    hypernet.add_task("A")
    weights = hypernet.weights_for("A")
    expected = dict(resnet_shaped.named_parameters())

    assert set(weights) == set(expected)
    for name, value in weights.items():
        assert value.shape == expected[name].shape, name

    assert hypernet.raw_for("A").numel() == sum(
        p.numel() for p in expected.values()
    )


def test_one_head_when_the_target_has_one_parameter_type():
    """Our plain CNN has no BatchNorm and no residuals, so one head is correct."""
    _, hypernet, _ = build()
    assert set(hypernet.heads) == {"weights"}


def test_snapshot_does_not_follow_the_live_network():
    uncle, hypernet, _ = build()
    hypernet.add_task("A")
    snapshot = hypernet.snapshot()

    code = hypernet.task_codes["A"].detach()
    before = snapshot.weights_from_code(code)["0.weight"].clone()

    with torch.no_grad():
        for parameter in hypernet.trunk.parameters():
            parameter.add_(0.1)

    after = snapshot.weights_from_code(code)["0.weight"]
    assert torch.equal(before, after), "the snapshot moved when the live network did"
    assert not any(p.requires_grad for p in snapshot.parameters())


def test_preserve_is_zero_with_nothing_to_protect():
    uncle, hypernet, _ = build()
    hypernet.add_task("A")
    assert uncle.preserve([], hypernet.snapshot()).item() == 0.0


def test_learn_then_forget():
    uncle, _, _ = build()

    uncle.learn("A", protected=[])
    learned_a = uncle.accuracy("A")
    assert learned_a > 70, f"task A did not learn, scored {learned_a:.1f}%"

    uncle.learn("B", protected=["A"])
    kept_a = uncle.accuracy("A")
    assert kept_a > 50, f"task A was wrecked by learning B, scored {kept_a:.1f}%"

    before_b = uncle.accuracy("B")
    uncle.forget("A", protected=["B"])

    forgotten_a = uncle.accuracy("A")
    assert forgotten_a < 30, f"task A was not forgotten, scored {forgotten_a:.1f}%"

    moved_b = abs(uncle.accuracy("B") - before_b)
    assert moved_b < 15, f"forgetting A spilled onto B by {moved_b:.1f} points"


def test_forget_needs_no_data():
    """The forget path must not read the task's dataset."""
    uncle, _, _ = build()
    uncle.learn("A", protected=[])

    uncle.tasks["A"]["train"] = None  # any read would raise
    uncle.forget("A", protected=[])


def test_metrics_match_the_paper_definitions():
    history = [
        {"index": 0, "action": "learn", "task": "A", "before": {},
         "after": {"A": 90.0}, "seen": ["A"], "forgotten": []},
        {"index": 1, "action": "forget", "task": "A",
         "before": {"A": 90.0, "B": 80.0}, "after": {"A": 10.0, "B": 78.0},
         "seen": ["A", "B"], "forgotten": ["A"]},
        {"index": 2, "action": "learn", "task": "C",
         "before": {"A": 10.0, "B": 78.0},
         "after": {"A": 14.0, "B": 77.0, "C": 85.0},
         "seen": ["A", "B", "C"], "forgotten": ["A"]},
    ]

    assert spill(history[1]) == 2.0            # only B moved, by 2 points
    assert spill(history[0]) is None           # learning has no spill
    assert relapse(history) == {"A": 4.0}      # 10 right after, 14 at the end

    numbers = summary(history)
    assert numbers["forget_accuracy"] == 14.0
    assert numbers["retain_accuracy"] == 81.0  # mean of B and C
    assert numbers["mean_spill"] == 2.0


def test_burn_in_anneals():
    """Appendix C: start at 100, drop 10% per unlearn, never below 20."""
    config = Config(
        tasks=("A",), requests=(("learn", "A"),), backbone="cnn", device="cpu",
    )
    assert config.burn_in_for(0) == 100
    assert config.burn_in_for(1) == 90
    assert config.burn_in_for(2) == 81
    assert config.burn_in_for(50) == config.burn_in_min


def test_sequences_parse_to_the_papers_requests():
    from uncle.config import PERMUTED_MNIST_SEQUENCES, parse_sequence

    requests = parse_sequence(PERMUTED_MNIST_SEQUENCES[1])
    assert len(requests) == 15
    assert requests[0] == ("learn", "1")
    assert requests[2] == ("forget", "1")
    assert sum(1 for action, _ in requests if action == "forget") == 5


def test_resnet_target_keeps_buffers_per_task():
    """BatchNorm statistics are not generated, so each task holds its own."""
    config = Config(
        tasks=("A", "B"), requests=(("learn", "A"), ("learn", "B")),
        backbone="resnet18", chunks=200, epochs=1, batch_size=8, device="cpu",
    )
    target = build_target(config)

    assert any("running_mean" in name for name, _ in target.named_buffers())

    hypernet = HyperNetwork(config=config, target=target)
    assert set(hypernet.heads) == {"weights", "batchnorm", "residual"}

    uncle = UnCLe(hypernet, config, target, tasks=None)
    assert uncle.buffer_template, "the template should hold ResNet's buffers"

    # Generation has to cover every ResNet parameter, exactly once.
    hypernet.add_task("A")
    weights = hypernet.weights_for("A")
    expected = dict(target.named_parameters())
    assert set(weights) == set(expected)
    for name, value in weights.items():
        assert value.shape == expected[name].shape, name

    # One real forward pass, with buffers supplied the way learn() does it.
    uncle.task_buffers["A"] = {
        name: value.clone() for name, value in uncle.buffer_template.items()
    }
    before = uncle.task_buffers["A"]["bn1.running_mean"].clone()

    target.train()
    scores = functional_call(
        target, (weights, uncle.task_buffers["A"]),
        (torch.randn(8, 1, 28, 28),), strict=True,
    )

    assert scores.shape == (8, 10)
    assert torch.isfinite(scores).all()
    # Train mode must have moved this task's statistics.
    assert not torch.equal(before, uncle.task_buffers["A"]["bn1.running_mean"])


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")

    print(f"\n{len(tests)} checks passed.")


if __name__ == "__main__":
    main()
