import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

spec = importlib.util.spec_from_file_location(
    "model_experiments",
    Path(__file__).resolve().parents[1] / "tools/model_experiments.py",
)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


class DatasetTests(unittest.TestCase):
    def make(self, root, role, group="g1", shade=150):
        root.mkdir()
        image = cv2.imencode(".png", np.full((100, 100, 3), shade, np.uint8))[
            1
        ].tobytes()
        (root / "image.png").write_bytes(image)
        a = {
            "id": "a1",
            "well_uid": "w1",
            "image_path": "image.png",
            "frame_index": 0,
            "eligible_for_training": True,
            "label": "coma",
            "label_source": "manual",
            "occupancy": "single",
            "boundary_normalized": [[0, 0], [1, 0], [1, 1], [0, 1]],
        }
        m = {
            "task": "ccrt",
            "role": role,
            "experiment_id": group,
            "images": [
                {
                    "path": "image.png",
                    "frame_number": 1,
                    "sha256": hashlib.sha256(image).hexdigest(),
                }
            ],
            "annotations": [a],
        }
        (root / "annotations.json").write_text(json.dumps(m))
        return E.Dataset(root, role)

    def test_holdouts_reject_same_group_and_same_pixels_even_with_changed_names(self):
        with tempfile.TemporaryDirectory() as out:
            root = Path(out)
            train = self.make(root / "train", "train")
            test = self.make(root / "test", "test", shade=160)
            with self.assertRaisesRegex(ValueError, "split group"):
                E.check_holdouts(train, [test])
            test = self.make(root / "duplicate", "test", "other", 150)
            with self.assertRaisesRegex(ValueError, "content hash"):
                E.check_holdouts(train, [test])
            test = self.make(root / "separate", "test", "other", 160)
            E.check_holdouts(train, [test])
            with self.assertRaisesRegex(ValueError, "Expected a train"):
                E.Dataset(test.root, "train")

    def test_plan_rejects_changed_manifest_and_partial_well_sample_selections(self):
        with tempfile.TemporaryDirectory() as out:
            root = Path(out)
            train = self.make(root / "train", "train")
            plan = {
                "kind": "flyscope-comparison-plan",
                "dataset_sha256": train.sha,
                "runs": [
                    {
                        "id": "one",
                        "seed": 77,
                        "well_uids": ["w1"],
                        "annotation_ids": ["a1"],
                    }
                ],
            }
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(
                E.load_plan(train, path)["runs"][0]["annotation_ids"], ["a1"]
            )
            bad = copy.deepcopy(plan)
            bad["runs"][0]["annotation_ids"] = ["missing"]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "exactly"):
                E.load_plan(train, path)
            bad = copy.deepcopy(plan)
            bad["dataset_sha256"] = "different"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "different dataset"):
                E.load_plan(train, path)

    def test_evaluation_scores_only_annotations_and_keeps_predictions_for_other_wells(
        self,
    ):
        with tempfile.TemporaryDirectory() as out:
            test = self.make(Path(out) / "test", "test")
            p = [[0, 0], [1, 0], [1, 1], [0, 1]]
            test.manifest["wells"] = [
                {"well_uid": uid, "frames": [{"frame_index": 0, "points": p}]}
                for uid in ["w1", "unannotated"]
            ]

            class Adapter:
                metadata = {"name": "contract test"}

                def predict(self, content, wells):
                    return [
                        {"well_id": w["id"], "prediction": "coma", "confidence": 0.7}
                        for w in wells
                    ]

            result = E.evaluate(Adapter(), test)
            self.assertEqual(result["scored_annotations"], 1)
            self.assertEqual(result["unscored_predictions"], 1)
            self.assertEqual(result["accuracy"], 1)
