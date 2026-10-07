import importlib.util
import json
import unittest
from pathlib import Path
import cv2
import numpy as np
from backend.services.keras_adapter import (
    KerasAdapter,
    MODELS,
    crop_well,
    preprocess,
    occupancy_features,
    fit_occupancy,
    occupancy_decision,
)
from backend.services.detection import detect_wells, _clean_single_polygon


class ModelTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow"),
        "Install requirements-models.txt for CNN checks",
    )
    def test_original_models_predict_real_probabilities_with_explicit_occupancy(self):
        image = np.full((250, 250, 3), 150, np.uint8)
        cv2.ellipse(image, (120, 120), (35, 12), 30, 0, 360, (30, 60, 100), -1)
        content = cv2.imencode(".png", image)[1].tobytes()
        well = {"id": "w1", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        for name in ["model1.keras", "model2.keras"]:
            adapter = KerasAdapter(MODELS / name)
            record = adapter.predict(content, [well])[0]
            self.assertIn(record["prediction"], ["coma", "awake"])
            self.assertAlmostEqual(
                sum(record["state_probabilities"].values()), 1, places=5
            )
            self.assertEqual(
                record["occupancy_method"], "foreground-screen-v1 (uncalibrated)"
            )
            blank = cv2.imencode(".png", np.full_like(image, 150))[1].tobytes()
            record = adapter.predict(blank, [well])[0]
            self.assertEqual(record["prediction"], "empty")
            self.assertIsNone(record["confidence"])
            partial = adapter.predict(blank, [{**well, "partial": True}])[0]
            self.assertEqual(partial["prediction"], "unknown")

    def test_preprocessing_is_same_for_full_masks_as_legacy_and_masks_exterior(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (250, 250, 3), np.uint8)
        adjusted = cv2.cvtColor(
            cv2.convertScaleAbs(image, alpha=1.5, beta=120), cv2.COLOR_BGR2GRAY
        )
        binary = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        expected = cv2.resize(binary, (200, 200)).astype(np.float32) / 255
        np.testing.assert_array_equal(
            preprocess(image, np.full((250, 250), 255, np.uint8))[..., 0], expected
        )
        crop, mask = crop_well(image, [[0, 0], [1, 0], [0.5, 1]])
        self.assertEqual(preprocess(crop, mask).shape, (200, 200, 1))
        self.assertEqual(preprocess(crop, mask)[-1, 0, 0], 1)

    def test_occupancy_head_learns_only_selected_features_and_has_separate_provenance(
        self,
    ):
        x = [
            [0, 0, 0, 0],
            [0.001, 0, 0.001, 0],
            [0.1, 0.1, 0.15, 0.1],
            [0.2, 0.1, 0.2, 0.1],
        ]
        head = fit_occupancy(x, [0, 0, 1, 1])
        self.assertEqual(head["training_examples"], 4)
        self.assertEqual(
            occupancy_decision(np.array(x[0]), "unknown", head)[0], "empty"
        )
        self.assertEqual(
            occupancy_decision(np.array(x[-1]), "unknown", head)[0], "present"
        )
        self.assertIsNone(fit_occupancy(x, [1, 1, 1, 1]))

    def test_single_detection_rejects_clumps_and_spiky_shapes(self):
        image = np.full((250, 250, 3), 150, np.uint8)
        cv2.circle(image, (125, 125), 15, (40, 40, 40), 3)
        for method in ["auto", "contours"]:
            self.assertEqual(
                detect_wells(
                    cv2.imencode(".png", image)[1].tobytes(),
                    method=method,
                    layout="individual",
                )["wells"],
                [],
            )
        square = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        self.assertTrue(_clean_single_polygon(square, 250, 250))
        clump = [[0.1, 0.1], [0.1001, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        cleaned = _clean_single_polygon(clump, 250, 250)
        self.assertEqual(len(cleaned), 4)
        spike = [
            [0.1, 0.1],
            [0.4, 0.1],
            [0.5, 0.8],
            [0.51, 0.1],
            [0.9, 0.1],
            [0.9, 0.9],
            [0.1, 0.9],
        ]
        self.assertEqual(_clean_single_polygon(spike, 250, 250), [])
