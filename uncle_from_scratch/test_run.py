"""One end-to-end check of the learn and forget loop on real images.

Small on purpose: the `cnn` backbone, 8 chunks, two tasks, and a few hundred
images each, so it runs on a CPU. It needs the extracted dataset in
`../data/tiny-imagenet-200`; it does not download anything.

What it asserts is the behaviour the method exists for: learning a task beats
chance, forgetting it returns to chance, and the other task survives.
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

from data import DEFAULT_ROOT
from run import make_config, run_experiment
from streams import build_tasks

CHANCE = 10.0


class ConfigTests(unittest.TestCase):
    def test_limit_requests_keeps_the_sequence_valid(self):
        config = make_config(sequence=1, limit_requests=5, backbone="cnn")
        self.assertEqual(len(config.requests), 5)
        self.assertEqual(config.beta, 0.01)          # the paper's Tiny ImageNet value
        self.assertEqual(len(config.tasks), 20)

    def test_overrides_reach_the_config(self):
        config = make_config(backbone="cnn", chunks=32, gamma=0.5, epochs=2)
        self.assertEqual((config.chunks, config.gamma, config.epochs), (32, 0.5, 2))


@unittest.skipUnless(DEFAULT_ROOT.exists(), f"No dataset at {DEFAULT_ROOT}")
class RunTests(unittest.TestCase):
    def test_learn_beats_chance_then_forget_returns_to_it(self):
        config = Config(
            dataset="tiny_imagenet",
            tasks=("0", "1"),
            requests=parse_sequence("L0 L1 U0"),
            backbone="cnn",
            chunks=8,
            epochs=12,
            batch_size=32,
            beta=0.01,
            seed=0,
            device="cpu",
        )
        torch.manual_seed(config.seed)
        history, numbers = run_experiment(config=config, max_images=300, verbose=False)

        self.assertEqual([record["action"] for record in history],
                         ["learn", "learn", "forget"])

        learned = history[0]["after"]["0"]
        forget = history[2]
        self.assertGreater(learned, CHANCE, "learning task 0 did not beat chance")
        self.assertLessEqual(forget["after"]["0"], CHANCE + 5.0,
                             "task 0 still works after being forgotten")
        self.assertGreater(forget["after"]["1"], CHANCE,
                           "forgetting task 0 destroyed task 1")

        self.assertAlmostEqual(numbers["retain_accuracy"], forget["after"]["1"])
        self.assertAlmostEqual(numbers["forget_accuracy"], forget["after"]["0"])
        self.assertIsNotNone(numbers["mean_spill"])

    def test_capping_keeps_every_class(self):
        tasks = build_tasks(include=["0"], max_images=300)
        train = tasks["0"]["train"]
        self.assertEqual(len(train), 300)
        labels = {train[index][1] for index in range(len(train))}
        self.assertEqual(labels, set(range(10)))


if __name__ == "__main__":
    unittest.main()
