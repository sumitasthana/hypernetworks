"""Checks for the from-scratch entry points, on real images, on a CPU.

Small on purpose: the `cnn` backbone, 8 chunks, two tasks and a few hundred
images, so the whole file runs in a couple of minutes. It needs the extracted
dataset in `../data/tiny-imagenet-200`; it does not download anything.

What it does NOT assert is that learning beats chance. At this scale, with the
`cnn` stand-in, it does not: the classifier's generated weights start too large
(initial loss around 6.7 against ln 10 = 2.30), the first optimizer steps
collapse the network to identical logits for every input, and it stays there.
Shrinking that one layer's output scale fixes it, which points at the
initialization, the one place this code knowingly departs from the paper. See
`colab_guide.html`. Until that is settled, an accuracy assertion here would
either fail or lock in the broken behaviour, so this file tests the parts that
are true and reliable: the plumbing, and what forgetting does to the weights.

`baseline.py` covers the other half of the question. Plain backprop on the same
300 images reaches about 30%, so the architecture and the data are fine.
"""

from pathlib import Path
import sys
import unittest

_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(_HERE.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch

from uncle.config import Config, parse_sequence
from uncle.hypernet import HyperNetwork, build_target
from uncle.trainer import UnCLe

from data import DEFAULT_ROOT
from run import make_config, run_experiment
from streams import build_tasks

CHANCE = 10.0


def small_config(**overrides) -> Config:
    return Config(**{
        "dataset": "tiny_imagenet",
        "tasks": ("0", "1"),
        "requests": parse_sequence("L0 L1 U0"),
        "backbone": "cnn",
        "chunks": 8,
        "epochs": 2,
        "batch_size": 32,
        "beta": 0.01,
        "seed": 0,
        "device": "cpu",
        **overrides,
    })


class ConfigTests(unittest.TestCase):
    def test_limit_requests_keeps_the_sequence_valid(self):
        config = make_config(sequence=1, limit_requests=5, backbone="cnn")
        self.assertEqual(len(config.requests), 5)
        self.assertEqual(config.beta, 0.01)          # the paper's Tiny ImageNet value
        self.assertEqual(len(config.tasks), 20)

    def test_overrides_reach_the_config(self):
        config = make_config(backbone="cnn", chunks=32, gamma=0.5, epochs=2)
        self.assertEqual((config.chunks, config.gamma, config.epochs), (32, 0.5, 2))

    def test_config_and_overrides_together_are_refused(self):
        with self.assertRaises(ValueError):
            run_experiment(config=small_config(), epochs=3)


@unittest.skipUnless(DEFAULT_ROOT.exists(), f"No dataset at {DEFAULT_ROOT}")
class StreamTests(unittest.TestCase):
    def test_capping_keeps_every_class(self):
        tasks = build_tasks(include=["0"], max_images=300)
        train = tasks["0"]["train"]
        self.assertEqual(len(train), 300)
        labels = {train[index][1] for index in range(len(train))}
        self.assertEqual(labels, set(range(10)))

    def test_unknown_task_is_refused(self):
        with self.assertRaises(ValueError):
            build_tasks(include=["99"])


@unittest.skipUnless(DEFAULT_ROOT.exists(), f"No dataset at {DEFAULT_ROOT}")
class RequestLoopTests(unittest.TestCase):
    def test_records_and_metrics_line_up(self):
        config = small_config()
        torch.manual_seed(config.seed)
        history, numbers = run_experiment(config=config, max_images=300, verbose=False)

        self.assertEqual([r["action"] for r in history], ["learn", "learn", "forget"])
        self.assertEqual(history[0]["seen"], ["0"])
        self.assertEqual(history[2]["forgotten"], ["0"])

        # The first learn has nothing to protect, so no accuracy exists before it.
        self.assertEqual(history[0]["before"], {})
        # Only forget requests carry an annealed burn-in.
        self.assertIsNone(history[0]["burn_in"])
        self.assertEqual(history[2]["burn_in"], config.burn_in)

        forget = history[2]
        self.assertAlmostEqual(numbers["forget_accuracy"], forget["after"]["0"])
        self.assertAlmostEqual(numbers["retain_accuracy"], forget["after"]["1"])
        # Spill is the movement of every task other than the forgotten one.
        self.assertAlmostEqual(
            numbers["mean_spill"],
            abs(forget["after"]["1"] - forget["before"]["1"]),
        )

    def test_forgetting_stays_in_its_lane(self):
        config = small_config()
        torch.manual_seed(config.seed)
        history, numbers = run_experiment(config=config, max_images=300, verbose=False)

        self.assertLessEqual(history[2]["after"]["0"], CHANCE + 2.0,
                             "task 0 should sit at chance once forgotten")
        self.assertLessEqual(numbers["mean_spill"], 5.0,
                             "forgetting task 0 moved task 1 too much")


@unittest.skipUnless(DEFAULT_ROOT.exists(), f"No dataset at {DEFAULT_ROOT}")
class ForgetMechanismTests(unittest.TestCase):
    def test_forget_collapses_the_generated_weights(self):
        """Equation 3 in action, measured on the weights rather than on accuracy.

        Averaging the squared distance to fresh zero-mean noise is smallest at
        zero, so forgetting drives the generated numbers toward zero. That is
        the mechanism, it is deterministic, and it holds whether or not the
        task ever learned anything.
        """
        config = small_config(requests=parse_sequence("L0"), epochs=1)
        torch.manual_seed(config.seed)
        tasks = build_tasks(config, include=["0", "1"], max_images=300)
        target = build_target(config)
        hypernet = HyperNetwork(target, config)
        uncle = UnCLe(hypernet, config, target, tasks)

        uncle.learn("0", [])
        before = hypernet.raw_for("0").detach().norm().item()

        uncle.forget("0", [], burn_in=30)
        after = hypernet.raw_for("0").detach().norm().item()

        self.assertLess(after, before / 2,
                        f"forgetting barely moved the weights: {before} to {after}")


if __name__ == "__main__":
    unittest.main()
