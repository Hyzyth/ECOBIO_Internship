"""Run in the shared app environment: python -m unittest tests.suite_contracts -v."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from backend.services.suite_metrics import metrics
from backend.services.suite_models import forest_export, forest_probabilities
from tools.model_suite import TrainingData, apply_policy, checked_plan, plan_for, main
from tests.suite_fixtures import grids


class SuiteContracts(unittest.TestCase):
    def test_partial_truth_is_not_a_negative_recovery(self):
        rows = []
        for i, truth in enumerate(["coma", None, None, None]):
            rows.append(
                {
                    "well_id": "w1",
                    "well_name": "W1",
                    "frame_index": i,
                    "frame_id": f"frame-{i}",
                    "filename": f"frame{i}.png",
                    "prediction": "coma" if i == 0 else "awake",
                    "raw_prediction": "awake",
                    "ground_truth": truth,
                    "scored": truth is not None,
                    "state_probabilities": {"coma": 0.1, "awake": 0.8, "empty": 0.1},
                }
            )
        m = metrics(rows, {"mode": "sequence"})
        self.assertEqual(m["overall"]["scored_annotations"], 1)
        self.assertEqual(m["overall"]["unscored_predictions"], 3)
        self.assertEqual(m["recovery"]["false_recoveries"], 0)
        self.assertIsNone(m["recovery"]["mae_frames"])
        for row in rows:
            row.update(ground_truth="coma", scored=True)
        self.assertEqual(
            metrics(rows, {"mode": "sequence"})["recovery"]["false_recoveries"], 1
        )
        self.assertEqual(metrics(rows, {"mode": "single"})["recovery"]["wells"], [])

    def test_streak_requires_contiguous_frames_and_retains_raw_scores(self):
        records = [
            {"frame_index": i, "prediction": "awake", "confidence": 0.95}
            for i in [3, 5, 6, 7, 8]
        ]
        records[-1].update(prediction="coma", confidence=0.99)
        apply_policy(
            records, {"confidence_threshold": 0.8, "awake_consecutive_frames": 3}
        )
        self.assertEqual(
            [r["prediction"] for r in records],
            ["unsure", "awake", "awake", "awake", "awake"],
        )
        self.assertEqual(records[-1]["raw_prediction"], "coma")
        self.assertEqual(records[-1]["raw_confidence"], 0.99)
        self.assertIsNone(records[-1]["confidence"])
        self.assertEqual(records[-1]["recovery_onset_frame_index"], 5)
        self.assertEqual(records[-1]["recovery_confirmed_frame_index"], 7)

    def test_partial_boundary_cannot_confirm_recovery(self):
        records = [
            {
                "frame_index": i,
                "prediction": "awake",
                "confidence": None,
                "raw_confidence": 0.99,
            }
            for i in range(3)
        ]
        apply_policy(
            records, {"confidence_threshold": 0.8, "awake_consecutive_frames": 2}
        )
        self.assertTrue(all(r["prediction"] == "unsure" for r in records))
        self.assertTrue(all(r["raw_confidence"] == 0.99 for r in records))

    def test_forests_use_portable_json_with_identical_probabilities(self):
        x = np.random.default_rng(2).normal(size=(20, 6))
        y = np.array([0, 1] * 10)
        forest = RandomForestClassifier(n_estimators=5, random_state=42).fit(x, y)
        artifact = json.loads(json.dumps(forest_export(forest), allow_nan=False))
        np.testing.assert_allclose(
            forest_probabilities(artifact, x),
            np.c_[forest.predict_proba(x), np.zeros(len(x))],
        )

    def test_selection_binding_and_causal_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, _ = grids(Path(temporary) / "fixtures")
            data = TrainingData([train])
            plan = plan_for(data, 42, "W1", "W2")
            self.assertEqual(plan, plan_for(data, 42, "W1", "W2"))
            self.assertEqual(len(plan["runs"]), 23)
            for run in plan["runs"]:
                if run["id"].startswith("random"):
                    self.assertTrue(all(data.occupied[w] for w in run["well_uids"]))
            bad = copy.deepcopy(plan)
            bad["runs"][0]["annotation_ids"].pop()
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "exactly"):
                checked_plan(data, path)
            identity = next(
                k for k, (_, a) in data.annotations.items() if a["frame_index"] == 2
            )
            calls = []
            data.rgb = lambda sha, frame, well, size: (
                calls.append(frame) or np.zeros((8, 8, 3)),
                None,
            )
            data.clip(identity, 8, 3)
            self.assertEqual(calls, [0, 1, 2])
            data.datasets[0].manifest["analysis_start_index"] = 2
            calls.clear()
            data.clip(identity, 8, 3)
            self.assertEqual(calls, [2, 2, 2])

    def test_training_grid_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, _ = grids(Path(temporary) / "fixtures")
            with self.assertRaisesRegex(ValueError, "Duplicate training"):
                TrainingData([train, train])

    def test_independent_inputs_rejected_before_any_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, test = grids(root / "fixtures")
            path = train / "annotations.json"
            manifest = json.loads(path.read_text())
            manifest["mode"] = "single"
            path.write_text(json.dumps(manifest))
            with patch("tools.model_suite.train_model") as trainer:
                with self.assertRaisesRegex(ValueError, "ordered multi-frame"):
                    main(
                        [
                            "compare",
                            "--train",
                            str(train),
                            "--test",
                            str(test),
                            "--output",
                            str(root / "results"),
                        ]
                    )
                trainer.assert_not_called()
                self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
