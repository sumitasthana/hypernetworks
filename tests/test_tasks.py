"""Small task-label and partition checks, without downloading images."""

import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncle.tasks import ClassTask, load_partition


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


if __name__ == "__main__":
    unittest.main()
