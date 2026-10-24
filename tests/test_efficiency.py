"""Efficiency options preserve inputs, manual truth and recovery semantics."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from collections import OrderedDict
import cv2
import numpy as np
from flyscope.imaging.foreground import isolate, ISOLATED
from flyscope.models.runtime import crop_rgb, make_resnet, model_version
from flyscope.models.recovery import RecoveryTracker, apply_policy
from flyscope.training.sampling import thin_targets
from flyscope.training.data import TrainingData
from flyscope.training.support import CropCache, Progress
from flyscope.reporting.inspection import inspect, resnet_maps
from tests.suite_fixtures import grids


class EfficiencyTests(unittest.TestCase):
    def test_isolation_keeps_legs_removes_separate_dust_and_preserves_source(self):
        image = np.full((128, 128, 3), 190, np.uint8)
        mask = np.full((128, 128), 255, np.uint8)
        cv2.ellipse(image, (60, 64), (12, 5), 0, 0, 360, (25, 25, 25), -1)
        cv2.line(image, (60, 64), (60, 74), (90, 90, 90), 1)
        cv2.circle(image, (100, 100), 2, (20, 20, 20), -1)
        original = image.copy()
        result, status = isolate(image, mask)
        self.assertEqual(status, "isolated")
        np.testing.assert_array_equal(result[74, 60], image[74, 60])
        self.assertTrue((result[100, 100] == 255).all())
        np.testing.assert_array_equal(image, original)
        empty = np.full_like(image, 190)
        np.testing.assert_array_equal(isolate(empty, mask)[0], empty)

    def test_isolated_crop_cache_matches_runtime_and_has_separate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, _ = grids(root / "data")
            data = TrainingData([train])
            ids = list(data.annotations)[:3]
            progress = Progress(root)
            caches = []
            try:
                for preprocessing in (data.preprocessing, ISOLATED):
                    data.preprocessing = preprocessing
                    data.rgb.cache_clear()
                    expected = {k: data.sample(k, 64) for k in ids}
                    cache = CropCache.prepare(
                        data, [{"annotation_ids": ids}], 64, 0, root / "cache", progress
                    )
                    caches.append(cache)
                    for k in ids:
                        dataset, a = data.annotations[k]
                        actual = cache.get(
                            dataset.sha, a["frame_index"], a["well_uid"], 64
                        )
                        np.testing.assert_array_equal(actual[0], expected[k][0])
                        np.testing.assert_array_equal(actual[1], expected[k][1])
                self.assertEqual(len(list((root / "cache").iterdir())), 2)
            finally:
                for cache in caches:
                    cache.close()
                progress.close()
                data.rgb.cache_clear()

    def test_thinning_keeps_leg_only_change_and_manual_transition_boundaries(self):
        base = np.full((32, 32, 3), 190, np.uint8)
        leg = base.copy()
        leg[14:18, 16] = 30
        annotations = {
            str(i): (
                SimpleNamespace(sha="dataset"),
                {
                    "frame_index": i,
                    "well_uid": "w",
                    "label": "coma" if i < 6 else "awake",
                },
            )
            for i in range(10)
        }
        data = SimpleNamespace(
            annotations=annotations,
            sample=lambda k, size: (base if int(k) < 2 else leg, None),
        )
        run = {"id": "one", "annotation_ids": list(annotations)}
        original = json.dumps(run)
        selected, report = thin_targets(data, run, 32, transition_radius=0)
        self.assertEqual(selected["annotation_ids"], ["0", "2", "5", "6", "9"])
        self.assertEqual(json.dumps(run), original)
        self.assertFalse(report["validation_thinned"])
        self.assertFalse(report["context_images_removed"])

    def test_static_runs_keep_spaced_anchors_and_dense_transition_window(self):
        image = np.full((32, 32, 3), 190, np.uint8)
        annotations = {
            str(i): (
                SimpleNamespace(sha="d"),
                {
                    "frame_index": i,
                    "well_uid": "w",
                    "label": "coma" if i < 80 else "awake",
                },
            )
            for i in range(160)
        }
        data = SimpleNamespace(
            annotations=annotations, sample=lambda k, size: (image, None)
        )
        run, _ = thin_targets(data, {"annotation_ids": list(annotations)}, 32)
        selected = [int(k) for k in run["annotation_ids"]]
        self.assertTrue(set(range(75, 86)) <= set(selected))
        self.assertLess(len(selected), 40)
        self.assertLessEqual(max(np.diff(selected)), 30)

    def test_temporal_runtime_clips_match_training_with_trimmed_start(self):
        import torch
        from flyscope.models.runtime import SuiteModel

        class Network:
            def features(self, clips):
                self.current = clips.detach().cpu().numpy()
                return torch.zeros((len(clips), 1024)), torch.zeros((len(clips), 3))

            def temporal(self, features):
                return torch.zeros((len(features), 3, features.shape[-1]))

        with tempfile.TemporaryDirectory() as tmp:
            train, _ = grids(Path(tmp))
            data = TrainingData([train])
            dataset = data.datasets[0]
            dataset.manifest["analysis_start_index"] = 2
            model = SuiteModel.__new__(SuiteModel)
            model.kind, model.size, model.device = (
                "deepethogram",
                64,
                torch.device("cpu"),
            )
            model.meta = {
                "window": 3,
                "preprocessing": data.preprocessing,
                "class_support": {},
            }
            model.sessions, model.network = OrderedDict(), Network()
            for frame in range(2, 6):
                identity = next(
                    k
                    for k, (_, a) in data.annotations.items()
                    if a["well_uid"] == "well-0" and a["frame_index"] == frame
                )
                geometry = data.context_geometry(dataset.sha, frame, "well-0")
                model.predict(
                    (train / data.image_maps[dataset.sha][frame]["path"]).read_bytes(),
                    [{"id": "well-0", "points": geometry["points"]}],
                    "s",
                    frame,
                )
                expected = (
                    np.stack(data.clip(identity, 64, 3)).transpose(0, 3, 1, 2) / 255
                )
                np.testing.assert_allclose(
                    model.network.current[0], expected, atol=1e-7
                )
            with self.assertRaisesRegex(ValueError, "contiguous"):
                model.predict(
                    (train / "images/frame-000006.png").read_bytes(),
                    [{"id": "well-0", "points": geometry["points"]}],
                    "s",
                    7,
                )

    def test_streaming_confirmation_matches_offline_policy_including_gaps(self):
        for automatic in (True, False):
            for require_coma in (True, False):
                for gap in (0, 2):
                    policy = dict(
                        confidence_threshold=0.8,
                        awake_consecutive_frames=3,
                        require_coma_before_awake=require_coma,
                        automatic_confirmation=automatic,
                    )
                    states = ["awake"] * 3 + ["coma", "awake", "awake", "awake", "coma"]
                    rows = [
                        {
                            "frame_index": i + (gap if i >= 6 else 0),
                            "prediction": state,
                            "confidence": 0.99,
                        }
                        for i, state in enumerate(states)
                    ]
                    tracker = RecoveryTracker(policy)
                    for r in rows:
                        tracker.observe(r, r["frame_index"])
                    apply_policy(rows, policy)
                    expected = next(
                        (
                            r["recovery_confirmed_frame_index"]
                            for r in rows
                            if r.get("recovery_confirmed_frame_index") is not None
                        ),
                        None,
                    )
                    self.assertEqual(tracker.confirmed, expected)

    def test_resnet_maps_have_probability_and_image_contracts_without_leaked_hooks(
        self,
    ):
        import torch

        torch.set_num_threads(2)
        network = make_resnet("scratch").eval()
        model = SimpleNamespace(
            kind="resnet18", network=network, device=torch.device("cpu")
        )
        rgb = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        probabilities, maps = resnet_maps(model, rgb)
        self.assertAlmostEqual(sum(probabilities.values()), 1, places=5)
        self.assertEqual(maps["gradcam-awake"].shape, rgb.shape)
        self.assertEqual(maps["layer1"].shape, (64, 256, 3))
        self.assertTrue(all(not module._forward_hooks for module in network.modules()))

    def test_resolution_report_uses_native_pixels_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, _ = grids(root / "data")
            report = inspect(train, root / "report", samples=4)
            self.assertEqual(len(report["samples"]), 4)
            self.assertEqual(report["native_side_pixels"]["median"], 130)
            self.assertGreater(report["samples"][0]["size_ratios"]["224"], 1)
            self.assertTrue((root / "report/index.html").is_file())

    def test_sampling_changes_model_identity_without_changing_legacy_versions(self):
        old = {"architecture": "resnet18"}
        version = model_version(old)
        self.assertNotEqual(
            version, model_version({**old, "training_sampling": {"minimum_pixels": 3}})
        )

    def test_pose_parent_crop_mismatch_is_rejected_before_inference(self):
        from flyscope.models.runtime import SuiteModel, CLASSES, PREPROCESSING, digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pose").mkdir()
            pose = root / "pose/pose.json"
            pose.write_text(json.dumps({"source_crop_preprocessing": ISOLATED}))
            meta = {
                "architecture": "sleap-rf",
                "image_size": 64,
                "classes": CLASSES,
                "preprocessing": PREPROCESSING,
                "artifact_hashes": {"pose/pose.json": digest(pose)},
            }
            meta["version"] = model_version(meta)
            path = root / "model.json"
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "different crop preprocessing"):
                SuiteModel(path)

    def test_evaluation_stopping_preserves_recovery_and_truth_denominators(self):
        from flyscope.evaluation import evaluate_model
        from flyscope.data.datasets import Dataset

        class Model:
            def __init__(self, *args):
                self.meta = dict(
                    architecture="resnet18",
                    training_dataset_sha256=[],
                    training_experiment_ids=[],
                    training_image_sha256=[],
                    selection={"id": "test"},
                    recovery_policy=dict(
                        confidence_threshold=0.8,
                        awake_consecutive_frames=1,
                        require_coma_before_awake=True,
                    ),
                )

            def predict(self, content, wells, session=None, frame=None):
                label = "coma" if frame < 2 else "awake"
                return [
                    dict(
                        well_id=w["id"],
                        prediction=label,
                        raw_prediction=label,
                        confidence=1.0,
                        state_probabilities={
                            k: float(k == label) for k in ("coma", "awake", "empty")
                        },
                    )
                    for w in wells
                ]

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            _, test = grids(Path(tmp))
            dataset = Dataset(test, "test")
            with patch("flyscope.evaluation.SuiteModel", Model):
                full = evaluate_model("unused", dataset)
                stopped = evaluate_model("unused", dataset, stop_after_awake=True)
            self.assertEqual(
                full["metrics"]["recovery"], stopped["metrics"]["recovery"]
            )
            self.assertEqual(
                full["metrics"]["baselines"], stopped["metrics"]["baselines"]
            )
            self.assertEqual(stopped["inference"]["skipped_well_frames"], 18)
            self.assertEqual(stopped["metrics"]["raw"]["scored_annotations"], 9)
            self.assertEqual(stopped["metrics"]["calibration"]["scored_predictions"], 9)
            self.assertEqual(stopped["metrics"]["overall"]["scored_annotations"], 18)


if __name__ == "__main__":
    unittest.main()
