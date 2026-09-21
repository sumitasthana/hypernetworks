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
`docs/colab_guide.html`. Until that is settled, an accuracy assertion here would
either fail or lock in the broken behaviour, so this file tests the parts that
are true and reliable: the plumbing, and what forgetting does to the weights.

`uncle/baseline.py` covers the other half of the question. Plain backprop on the same
300 images reaches about 30%, so the architecture and the data are fine.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from uncle.config import Config, parse_sequence
from uncle.hypernet import HyperNetwork, build_target
from uncle.trainer import UnCLe

from uncle.experiments import describe_costs, make_config, run_experiment
from uncle.telemetry import RunLog, environment
from uncle.streams import build_tasks
from uncle.tinyimagenet import DEFAULT_ROOT

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


class TelemetryTests(unittest.TestCase):
    """The cost log, on made-up rows, so this needs no GPU and no training."""

    def rows(self):
        log = RunLog(device=torch.device("cpu"))
        log.start()
        for index, (action, task) in enumerate([("learn", "0"), ("learn", "1"),
                                                ("forget", "0")]):
            log.measure({"index": index, "action": action, "task": task,
                         "seen": ["0", "1"][:index + 1]},
                        steps=40 if action == "learn" else 100)
        return log

    def test_protected_count_excludes_the_requests_own_task(self):
        """The first learn protects nothing, however the record counts tasks."""
        rows = self.rows().rows
        self.assertEqual([row["protected"] for row in rows], [0, 1, 1])

    def test_totals_separate_learning_from_forgetting(self):
        totals = self.rows().totals()
        self.assertEqual(totals["requests"], 3)
        self.assertEqual(totals["learn"]["requests"], 2)
        self.assertEqual(totals["forget"]["requests"], 1)
        self.assertGreaterEqual(totals["learn"]["slowest_seconds"],
                                totals["learn"]["mean_seconds"])
        # The key is always present so callers need not branch on hardware;
        # None is what "no GPU" looks like.
        self.assertIn("peak_memory_bytes", totals)
        self.assertIsNone(totals["peak_memory_bytes"])

    def test_the_printed_forms_hold_together(self):
        log = self.rows()
        table = log.table()
        self.assertEqual(len(table.splitlines()), 5)     # header, rule, three rows
        self.assertIn("forget", table)
        self.assertIn("total time", describe_costs(log.totals()))

    def test_environment_records_what_a_number_needs_to_be_traced(self):
        facts = environment(small_config())
        for key in ("torch", "device", "config", "platform"):
            self.assertIn(key, facts)
        self.assertEqual(facts["config"]["backbone"], "cnn")
        self.assertEqual(facts["config"]["chunks"], 8)


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
        result = run_experiment(config=config, max_images=300,
                                verbose=False, progress=False)
        history, numbers = result["history"], result["numbers"]

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

    def test_before_reuses_the_previous_after(self):
        """Nothing touches the model between requests, so re-measuring is waste."""
        config = small_config()
        torch.manual_seed(config.seed)
        history = run_experiment(config=config, max_images=300,
                                 verbose=False, progress=False)["history"]

        self.assertEqual(history[0]["before"], {})
        for index in range(1, len(history)):
            self.assertEqual(history[index]["before"], history[index - 1]["after"])

    def test_forgetting_stays_in_its_lane(self):
        config = small_config()
        torch.manual_seed(config.seed)
        result = run_experiment(config=config, max_images=300,
                                verbose=False, progress=False)
        history, numbers = result["history"], result["numbers"]

        self.assertLessEqual(history[2]["after"]["0"], CHANCE + 2.0,
                             "task 0 should sit at chance once forgotten")
        self.assertLessEqual(numbers["mean_spill"], 5.0,
                             "forgetting task 0 moved task 1 too much")

    def test_learning_actually_moves_off_chance(self):
        """The check that the classifier's output scale exists to make possible.

        With He initialization on the classifier this could not pass: the loss
        pinned to ln(10) = 2.3026 and accuracy to exactly 10.0, however many
        epochs it was given. Twenty epochs on 300 images is a small budget, so
        the bar is the loss falling clearly below ln(10) rather than a
        particular accuracy, which is still noisy at this scale.
        """
        config = small_config(requests=parse_sequence("L0"), tasks=("0",), epochs=20)
        torch.manual_seed(config.seed)
        history = run_experiment(config=config, max_images=300,
                                 verbose=False, progress=False)["history"]

        self.assertLess(history[0]["final_loss"], 2.0,
                        "training loss never fell below chance-level cross-entropy")
        self.assertGreater(history[0]["after"]["0"], CHANCE,
                           "learning task 0 did not beat chance")


class Killed(Exception):
    """Stands in for a runtime that goes away mid-run."""


@unittest.skipUnless(DEFAULT_ROOT.exists(), f"No dataset at {DEFAULT_ROOT}")
class CheckpointTests(unittest.TestCase):
    def config(self):
        return small_config(tasks=("0", "3", "9"),
                            requests=parse_sequence("L3 L0 U3 L9"), epochs=1)

    def run_to(self, output, stop_after=None, **kwargs):
        def die(record):
            if record["index"] == stop_after:
                raise Killed

        torch.manual_seed(0)
        return run_experiment(config=self.config(), max_images=100, output=output,
                              on_request=die if stop_after is not None else None,
                              verbose=False, progress=False, **kwargs)

    def test_a_resumed_run_matches_one_that_was_never_stopped(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            whole = self.run_to(a)

            with self.assertRaises(Killed):
                self.run_to(b, stop_after=1)

            # A different seed here must lose to the checkpoint's saved state.
            torch.manual_seed(999)
            resumed = run_experiment(config=self.config(), max_images=100,
                                     output=b, verbose=False, progress=False)

        self.assertEqual(len(resumed["history"]), 4)
        self.assertEqual(resumed["history"], whole["history"])
        self.assertEqual(resumed["numbers"], whole["numbers"])
        # The timings from before the restart come back too.
        self.assertEqual(len(resumed["costs"]), 4)

    def test_the_file_on_disk_keeps_the_requests_from_before_the_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Killed):
                self.run_to(tmp, stop_after=1)
            run_experiment(config=self.config(), max_images=100, output=tmp,
                           verbose=False, progress=False)
            saved = json.loads(
                (Path(tmp) / "history_seq1_cnn_seed0.json").read_text())
        self.assertEqual(len(saved), 4)

    def test_a_checkpoint_of_a_different_run_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Killed):
                self.run_to(tmp, stop_after=1)
            # The sequence, backbone and seed are all in the file name, so
            # those can never collide. The request list can: same name, a
            # different run. That is the case the guard is for.
            other = replace(self.config(),
                            requests=parse_sequence("L3 L0 U3"))
            with self.assertRaises(ValueError) as caught:
                run_experiment(config=other, max_images=100, output=tmp,
                               verbose=False, progress=False)
        self.assertIn("different run", str(caught.exception))

    def test_it_can_be_turned_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_to(tmp, checkpoint=False)
            self.assertFalse((Path(tmp) / "checkpoint_seq1_cnn_seed0.pt").exists())


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
