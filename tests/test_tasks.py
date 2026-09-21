"""Small task-label and partition checks, without downloading images."""

import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch

from uncle.tasks import ClassTask, load_partition
from uncle.tinyimagenet import missing_parts, prepare_data


class FakeDataset:
    class_to_idx = {"a": 0, "b": 1, "c": 2}
    class_names = ["apple", "bird", "cat"]

    def __init__(self, targets):
        self.targets = targets

    def __getitem__(self, index):
        return f"image-{index}", self.targets[index]


class TaskTests(unittest.TestCase):
    def test_nonidentity_mapping_and_split_order(self):
        train = ClassTask(FakeDataset([0, 1, 2, 1, 0]), ["c", "a"], 1)
        val = ClassTask(FakeDataset([2, 0, 1]), ["c", "a"], 1)
        self.assertEqual(train.indices, [0, 2, 4])
        self.assertEqual(train.targets, [1, 0, 1])
        self.assertEqual(train[1], ("image-2", 0))
        self.assertEqual(val.targets, [0, 1])
        self.assertEqual(train.class_to_idx, val.class_to_idx)
        self.assertEqual(train.class_names, ["cat", "apple"])

    def test_missing_or_duplicate_classes_fail(self):
        with self.assertRaises(ValueError):
            ClassTask(FakeDataset([0]), ["a", "c"], 1)
        with self.assertRaises(ValueError):
            ClassTask(FakeDataset([0]), ["a", "a"], 1)

    def test_partition_is_complete_disjoint_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "val").mkdir()
            (root / "val" / "val_annotations.txt").touch()
            (root / "words.txt").touch()
            classes = [f"n{i:08d}" for i in range(200)]
            (root / "wnids.txt").write_text("\n".join(reversed(classes)))
            path = root / "partition.json"
            partition = load_partition(root, path)
            before = path.read_bytes()
            self.assertEqual(partition, load_partition(root, path))
            self.assertEqual(before, path.read_bytes())
            flattened = [wnid for task in partition["tasks"] for wnid in task["wnids"]]
            self.assertEqual(sorted(flattened), classes)
            self.assertTrue(all(len(task["wnids"]) == 10 for task in partition["tasks"]))
            self.assertEqual(partition["tasks"][1]["task_id"], 1)
            with self.assertRaises(ValueError):
                load_partition(root, path, seed=43)
            self.assertEqual(before, path.read_bytes())
            edited = json.loads(before)
            edited["tasks"][1]["wnids"].reverse()
            path.write_text(json.dumps(edited))
            with self.assertRaises(ValueError):
                load_partition(root, path)


class GuideTests(unittest.TestCase):
    """The guide's code has to at least parse. Running it is check_guide.py."""

    def test_every_python_block_in_the_guide_parses(self):
        import ast
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from check_guide import COLAB_ONLY, SAMPLES, guide_blocks

        blocks = guide_blocks()
        self.assertGreater(len(blocks), 15)
        for index, block in enumerate(blocks):
            if index in SAMPLES:
                continue
            with self.subTest(block=index):
                ast.parse(block)      # COLAB_ONLY blocks are valid Python too


class IndexingTests(unittest.TestCase):
    """Only the requested classes get indexed, and labels stay global."""

    def setUp(self):
        from uncle.tinyimagenet import DEFAULT_ROOT
        if not DEFAULT_ROOT.exists():
            self.skipTest(f"No dataset at {DEFAULT_ROOT}")

    def test_a_subset_indexes_less_but_labels_do_not_move(self):
        from uncle.tinyimagenet import TinyImageNet
        everything = TinyImageNet(split="val")
        few = TinyImageNet(split="val", classes=everything.classes[:10])

        self.assertLess(len(few), len(everything))
        self.assertEqual(len(few.indexed_classes), 10)
        # Global numbering is unchanged, which is what ClassTask relies on.
        self.assertEqual(few.class_to_idx, everything.class_to_idx)
        self.assertEqual(set(few.targets), set(range(10)))

    def test_unknown_classes_are_refused(self):
        from uncle.tinyimagenet import TinyImageNet
        with self.assertRaises(ValueError):
            TinyImageNet(split="val", classes=["not-a-wnid"])


class PrepareDataTests(unittest.TestCase):
    """A directory that exists but is incomplete still needs downloading."""

    def test_a_partial_extraction_is_repaired_not_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tiny-imagenet-200"
            (root / "val").mkdir(parents=True)
            (root / "wnids.txt").touch()          # a stalled extraction
            self.assertTrue(missing_parts(root))

            def finish(url, target):
                (root / "words.txt").touch()
                (root / "train").mkdir(exist_ok=True)
                (root / "val" / "val_annotations.txt").touch()

            with patch("uncle.tinyimagenet.download_and_extract_archive", finish):
                self.assertEqual(prepare_data(root, download=True), root.resolve())

    def test_a_download_that_fixes_nothing_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tiny-imagenet-200"
            root.mkdir()
            with patch("uncle.tinyimagenet.download_and_extract_archive",
                       lambda url, target: None):
                with self.assertRaises(FileNotFoundError) as caught:
                    prepare_data(root, download=True)
            # Not "pass download=True", which is what the caller just did.
            self.assertIn("extraction was interrupted", str(caught.exception))
            self.assertNotIn("pass download=True", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
