"""Checkpoint-based diagnostics on synthetic data; no downloads required."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import TensorDataset

from uncle import Config, HyperNetwork, UnCLe, build_target, diagnose_forgetting
from uncle import checkpoint


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "before.pt"
        self.config = Config(
            tasks=("3", "0"), requests=(("learn", "3"), ("learn", "0")),
            backbone="cnn", hidden=(4,), chunks=4, epochs=1, batch_size=4,
            eval_batch_size=4, burn_in=3, noise_samples=2, device="cpu",
        )
        torch.manual_seed(14)
        self.tasks = {
            name: {"test": TensorDataset(torch.randn(8, 1, 28, 28), torch.arange(8))}
            for name in self.config.tasks
        }
        for splits in self.tasks.values():
            splits["train"] = splits["test"]
        uncle = self.fresh()
        for name in self.config.tasks:
            uncle.hypernet.add_task(name)
            uncle.task_buffers[name] = {}
        before = {name: uncle.accuracy(name) for name in self.config.tasks}
        history = [{"action": "learn", "task": name, "after": before}
                   for name in self.config.tasks]
        checkpoint.save(
            self.path, config=self.config, hypernet=uncle.hypernet, uncle=uncle,
            history=history, seen=list(self.config.tasks), forgotten=[],
            previous=before, costs=[], setup_seconds=0,
        )

    def fresh(self, config=None):
        config = self.config if config is None else config
        target = build_target(config)
        return UnCLe(HyperNetwork(target, config), config, target, self.tasks)

    def restored(self, config=None):
        uncle = self.fresh(config)
        checkpoint.restore(checkpoint.load(self.path), hypernet=uncle.hypernet, uncle=uncle)
        return uncle

    def diagnose(self, **kwargs):
        return diagnose_forgetting(self.path, tasks=self.tasks, verbose=False, **kwargs)

    def test_observer_does_not_change_updates_rng_or_modes(self):
        plain = self.restored()
        expected_losses = plain.forget("3", ["0"], burn_in=3)
        expected_rng = torch.get_rng_state().clone()
        observed = self.restored()
        rows = []

        def observe(record):
            rows.append(record)
            # Deliberately consume randomness and change modes like evaluation.
            torch.rand(7)
            observed.accuracy("0")
            observed.hypernet.eval()

        actual_losses = observed.forget("3", ["0"], burn_in=3, on_step=observe)
        self.assertEqual(expected_losses, actual_losses)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for name, value in plain.hypernet.state_dict().items():
            self.assertTrue(torch.equal(value, observed.hypernet.state_dict()[name]), name)
        self.assertTrue(observed.hypernet.training)
        self.assertEqual([r["step"] for r in rows], [0, 1, 2, 3])
        self.assertEqual(rows[1]["preserve_before"], 0.0)
        self.assertEqual(rows[1]["total_loss_before"], actual_losses[0])

    def test_diagnostic_restores_each_call_and_does_not_train_or_write_checkpoint(self):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch.object(UnCLe, "learn", side_effect=AssertionError("Must not train")):
            first = self.diagnose(steps=3, forgetting_lr=1e-5, gamma=1e-4)
            second = self.diagnose(steps=3, forgetting_lr=1e-5, gamma=1e-4)
        self.assertEqual(first["trace"], second["trace"])
        self.assertNotEqual(first["report_path"], second["report_path"])
        self.assertEqual(first["status"], "complete")
        self.assertTrue(first["initial_matches_checkpoint"])
        self.assertEqual(first["training_config"]["learning_rate"], self.config.learning_rate)
        self.assertEqual(first["settings"]["forgetting_lr"], 1e-5)
        self.assertEqual(json.loads(Path(first["report_path"]).read_text()),
                         json.loads(json.dumps(first)))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), digest)

    def test_legacy_checkpoint_loads_but_explicit_rate_mismatch_is_rejected(self):
        saved = checkpoint.load(self.path)
        del saved["config"]["forgetting_learning_rate"]
        torch.save(saved, self.path)
        loaded = checkpoint.load(self.path, self.config)
        self.assertIsNone(loaded["config"]["forgetting_learning_rate"])
        with self.assertRaisesRegex(ValueError, "different run"):
            checkpoint.load(self.path, replace(self.config, forgetting_learning_rate=1e-5))

    def test_forgetting_rate_changes_updates_without_changing_training_rate(self):
        fast = self.restored()
        fast.forget("3", ["0"], burn_in=1)
        slower = self.restored(replace(self.config, forgetting_learning_rate=1e-5))
        slower.forget("3", ["0"], burn_in=1)
        self.assertEqual(slower.config.learning_rate, fast.config.learning_rate)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            fast.hypernet.generator_parameters(), slower.hypernet.generator_parameters())))

    def test_forgetting_rate_does_not_change_learning(self):
        torch.manual_seed(123)
        first = self.fresh()
        expected = first.learn("3", [])
        torch.manual_seed(123)
        second = self.fresh(replace(self.config, forgetting_learning_rate=1e-5))
        actual = second.learn("3", [])
        self.assertEqual(actual, expected)
        for name, value in first.hypernet.state_dict().items():
            self.assertTrue(torch.equal(value, second.hypernet.state_dict()[name]), name)

    def test_cpu_rng_restoration_and_invalid_cuda_entries(self):
        saved = checkpoint.load(self.path)
        torch.set_rng_state(saved["rng"])
        expected = torch.rand(5)
        uncle = self.fresh()
        checkpoint.restore(saved, hypernet=uncle.hypernet, uncle=uncle)
        self.assertTrue(torch.equal(torch.rand(5), expected))
        saved["cuda_rng"] = [None]
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(ValueError, "CUDA device 0 RNG"):
                checkpoint.restore(saved, hypernet=uncle.hypernet, uncle=uncle)
        saved["cuda_rng"] = [torch.zeros(3, dtype=torch.uint8)]
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "device count"):
                checkpoint.restore(saved, hypernet=uncle.hypernet, uncle=uncle)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU RNG regression")
    def test_gpu_rng_tensors_are_restored_as_cpu_bytes(self):
        saved = checkpoint.load(self.path)
        saved["cuda_rng"] = [state.cuda() for state in torch.cuda.get_rng_state_all()]
        uncle = self.fresh(replace(self.config, device="cuda"))
        checkpoint.restore(saved, hypernet=uncle.hypernet, uncle=uncle)
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), saved["cuda_rng"][0].cpu()))

    def test_bad_arguments_and_unknown_or_forgotten_tasks(self):
        for kwargs in ({"steps": 0}, {"steps": True}, {"gamma": float("nan")},
                       {"forgetting_lr": 0}, {"task": "missing"}):
            with self.assertRaises(ValueError):
                self.diagnose(**kwargs)
        saved = checkpoint.load(self.path)
        saved["forgotten"] = ["3"]
        torch.save(saved, self.path)
        with self.assertRaisesRegex(ValueError, "already forgotten"):
            self.diagnose()

    def test_failure_keeps_partial_report(self):
        original = UnCLe.accuracy
        count = 0

        def interrupted(uncle, task):
            nonlocal count
            count += 1
            if count == 3:
                raise KeyboardInterrupt()
            return original(uncle, task)

        with patch.object(UnCLe, "accuracy", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.diagnose(steps=3)
        reports = list((self.root / "diagnostics").glob("*.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(len(report["trace"]), 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
