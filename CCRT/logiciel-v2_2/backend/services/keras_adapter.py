"""Legacy coma/awake networks, shared pixel preprocessing and occupancy screening.

The supplied weights have TWO classes. The foreground screen is an explicitly
uncalibrated rule, not an Empty class learned by those networks. Fine-tuning can
fit a separate occupancy head from manual empty/single-well examples.
"""

import hashlib
import json
import os
from pathlib import Path
from threading import Lock
import cv2
import numpy as np

PREPROCESSING = "legacy-contrast-otsu-200-v1-polygon-background"
CLASSES = {"coma": 0, "awake": 1}
MODELS = Path(__file__).resolve().parents[1] / "models"


def tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-models.txt to run or fine-tune the CNNs."
        ) from exc
    return tf


def crop_well(image, points):
    h, w = image.shape[:2]
    polygon = np.asarray(points, dtype=np.float64) * [w, h]
    left, top = np.maximum(0, np.floor(polygon.min(axis=0))).astype(int)
    right, bottom = np.minimum([w, h], np.ceil(polygon.max(axis=0))).astype(int)
    if right <= left or bottom <= top:
        raise ValueError("Well has no visible pixels.")
    crop = image[top:bottom, left:right].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon - [left, top]).astype(np.int32)], 255)
    if not mask.any():
        raise ValueError("Well has no visible pixels.")
    return crop, mask


def preprocess(crop, mask):
    # Original cnn.py: convertScaleAbs(alpha=1.5,beta=120), BGR->gray,
    # Otsu, resize 200x200, /255, one channel. Fill excluded pixels with
    # the interior median BEFORE Otsu so polygon masks don't change its histogram.
    adjusted = cv2.convertScaleAbs(crop, alpha=1.5, beta=120)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    gray[mask == 0] = np.median(gray[mask > 0])
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary[mask == 0] = 255
    return (cv2.resize(binary, (200, 200)).astype(np.float32) / 255.0)[..., None]


def occupancy_features(crop, mask):
    # Remove the bright rim by eroding in proportion to the visible crop.
    radius = max(1, round(min(mask.shape) * 0.04))
    interior = cv2.erode(mask, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)) > 0
    if interior.sum() < 25:
        return np.zeros(4), "unknown"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    median = np.median(gray[interior])
    mad = np.median(np.abs(gray[interior] - median))
    dark = interior & (gray < median - max(25, 4 * mad))
    colorful = interior & (hsv[..., 1] > 65) & (hsv[..., 2] < 230)

    def largest(binary):
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
        return (
            float(stats[1:, cv2.CC_STAT_AREA].max() / interior.sum()) if n > 1 else 0.0
        )

    feature = np.array(
        [
            largest(dark),
            largest(colorful),
            dark[interior].mean(),
            colorful[interior].mean(),
        ]
    )
    if feature[0] >= 0.012 or feature[1] >= 0.006:
        decision = "present"
    elif feature[0] < 0.002 and feature[1] < 0.002:
        decision = "empty"
    else:
        decision = "unknown"
    return feature, decision


def occupancy_decision(features, fallback, head=None):
    if not head:
        return fallback, None, "foreground-screen-v1 (uncalibrated)"
    standardized = (features - np.asarray(head["mean"])) / np.asarray(head["scale"])
    score = float(np.dot(standardized, head["weights"]) + head["bias"])
    probability = float(1 / (1 + np.exp(-np.clip(score, -40, 40))))
    # Probability estimates are uncalibrated; report them as scores explicitly.
    return (
        ("present" if probability >= 0.5 else "empty"),
        max(probability, 1 - probability),
        "learned-occupancy-v1",
    )


def fit_occupancy(features, present):
    if len(set(present)) < 2:
        return None
    x = np.asarray(features)
    y = np.asarray(present, float)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.001)
    x = (x - mean) / scale
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-np.clip(x @ weights + bias, -40, 40)))
        weights -= 0.08 * (x.T @ (p - y) / len(y) + 0.01 * weights)
        bias -= 0.08 * (p - y).mean()
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "training_examples": len(y),
    }


class KerasAdapter:
    def __init__(self, path, name=None, sidecar=None):
        self.path = Path(path)
        self.sidecar = sidecar or {}
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        version = hashlib.sha256(
            (
                digest
                + json.dumps(self.sidecar.get("occupancy_head"), sort_keys=True)
                + PREPROCESSING
            ).encode()
        ).hexdigest()
        self.metadata = {
            "name": name or self.path.stem,
            "version": version,
            "artifact_sha256": digest,
            "task": "ccrt",
            "classes": list(CLASSES),
            "preprocessing": PREPROCESSING,
            "empty_method": (
                "learned-occupancy-v1"
                if self.sidecar.get("occupancy_head")
                else "foreground-screen-v1 (uncalibrated)"
            ),
            "description": "Coma/awake CNN with separate occupancy handling. Original weights do not contain an Empty class. Review predictions before using them as measurements.",
        }
        self.model = None
        self.lock = Lock()

    def load(self):
        if self.model is None:
            self.model = tensorflow().keras.models.load_model(
                self.path, compile=False, safe_mode=True
            )
            if (
                tuple(self.model.input_shape[1:]) != (200, 200, 1)
                or self.model.output_shape[-1] != 2
            ):
                self.model = None
                raise RuntimeError("Unsupported model input/output contract.")
        return self.model

    def predict(self, image_bytes, wells):
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The selected file is not a readable image.")
        inputs, occupancies = [], []
        for well in wells:
            crop, mask = crop_well(image, well["points"])
            feature, fallback = occupancy_features(crop, mask)
            occupancies.append(
                occupancy_decision(
                    feature, fallback, self.sidecar.get("occupancy_head")
                )
            )
            inputs.append(preprocess(crop, mask))
        with self.lock:
            model = self.load()
            probabilities = np.concatenate(
                [
                    np.asarray(model(np.asarray(inputs[i : i + 8]), training=False))
                    for i in range(0, len(inputs), 8)
                ]
            )
        if (
            probabilities.shape != (len(wells), 2)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=0.001)
        ):
            raise RuntimeError("Model did not return valid class probabilities.")
        records = []
        for well, probability, occupancy in zip(wells, probabilities, occupancies):
            state, score, method = occupancy
            # Never mark an apparently empty partially visible well empty.
            if well.get("partial") and state == "empty":
                state = "unknown"
            label = (
                "empty"
                if state == "empty"
                else (
                    list(CLASSES)[int(np.argmax(probability))]
                    if state == "present"
                    else "unknown"
                )
            )
            confidence = float(probability.max()) if state == "present" else score
            records.append(
                {
                    "well_id": well["id"],
                    "prediction": label,
                    "confidence": confidence,
                    "state_probabilities": {
                        name: float(probability[i]) for name, i in CLASSES.items()
                    },
                    "occupancy_prediction": state,
                    "occupancy_score": score,
                    "occupancy_method": method,
                }
            )
        return records


def registered_adapters():
    adapters = {}
    for n in (1, 2):
        path = MODELS / f"model{n}.keras"
        if path.exists():
            adapters[f"legacy-model{n}"] = KerasAdapter(path, f"Original CNN {n}")
    # Only explicitly configured output directories are scanned. Trained artifacts
    # require a sidecar matching the exact file hash and preprocessing contract.
    directory = os.environ.get("FLYSCOPE_MODELS_DIR")
    if directory:
        for descriptor in sorted(Path(directory).rglob("*.json")):
            value = json.loads(descriptor.read_text())
            path = descriptor.with_suffix(".keras")
            if (
                not isinstance(value, dict)
                or value.get("kind") != "flyscope-model"
                or not path.exists()
            ):
                continue
            if value.get("preprocessing") != PREPROCESSING or hashlib.sha256(
                path.read_bytes()
            ).hexdigest() != value.get("artifact_sha256"):
                raise ValueError(
                    "Trained model descriptor does not match its artifact."
                )
            adapter = KerasAdapter(path, value.get("name"), value)
            adapters[f"trained-{adapter.metadata['version'][:12]}"] = adapter
    return adapters
