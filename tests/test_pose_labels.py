"""Anatomical data remain manual, hash-verified and grouped by whole fly."""

import json
from pathlib import Path
import tempfile
import unittest
from tests.suite_fixtures import grids
from flyscope.data.pose_labels import prepare, load_examples
from flyscope.training.pose_quality import split_examples


class PoseLabels(unittest.TestCase):
    def test_partial_labels_and_whole_individual_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, _ = grids(root / "fixture")
            m = prepare(train, root / "labels", count=12, size=64, wells=["W1", "W2"])
            self.assertEqual(
                {e["state_context"] for e in m["examples"]}, {"coma", "awake"}
            )
            with self.assertRaisesRegex(ValueError, "No reviewed"):
                load_examples(root / "labels/pose-labels.json", 64)
            for e in m["examples"][:6]:
                e["reviewed"] = True
                e["points"] = [
                    dict(status="visible", x=0.4, y=0.6),
                    dict(status="visible", x=0.6, y=0.4),
                ] + [dict(status="occluded", x=None, y=None)] * (len(m["nodes"]) - 2)
            p = root / "labels/pose-labels.json"
            p.write_text(json.dumps(m))
            nodes, examples, _ = load_examples(p, 64)
            self.assertEqual(len(examples), 6)
            self.assertEqual(examples[0][2].sum(), 2)
            fit, val, split = split_examples(examples)
            self.assertEqual(split["unit"], "whole individual")
            self.assertFalse(
                {e[3]["group_id"] for e in fit} & {e[3]["group_id"] for e in val}
            )
            with self.assertRaisesRegex(ValueError, "two independently"):
                split_examples(
                    [
                        e
                        for e in examples
                        if e[3]["group_id"] == examples[0][3]["group_id"]
                    ]
                )
            image = root / "labels" / m["examples"][0]["image_path"]
            image.write_bytes(image.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_examples(p, 64)

    def test_test_grids_cannot_prepare_training_keypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, test = grids(root / "fixture")
            with self.assertRaisesRegex(ValueError, "Expected a train"):
                prepare(test, root / "labels")
