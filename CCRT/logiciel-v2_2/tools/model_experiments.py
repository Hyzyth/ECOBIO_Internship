#!/usr/bin/env python3
"""Train/fine-tune and compare all registered CCRT networks with fixed selections.

Datasets are extracted app ZIPs. Test data are never used for training, fitting
occupancy, early stopping or selecting wells. Run from any working directory.
"""

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.keras_adapter import (
    KerasAdapter,
    CLASSES,
    PREPROCESSING,
    crop_well,
    preprocess,
    occupancy_features,
    fit_occupancy,
    tensorflow,
)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Dataset image path leaves dataset directory or is missing.")
    return path


class Dataset:
    def __init__(self, root, role):
        self.root = Path(root).resolve()
        raw = (self.root / "annotations.json").read_bytes()
        self.sha = digest(raw)
        self.manifest = json.loads(raw)
        if self.manifest.get("role", "train") != role:
            raise ValueError(f"Expected a {role} dataset, not another role.")
        if self.manifest.get("task") != "ccrt":
            raise ValueError(
                "These trainers support CCRT; egg format remains provisional."
            )
        self.role = role
        self.images = {image["path"]: image for image in self.manifest["images"]}
        self.image_hashes = set()
        for relative, image in self.images.items():
            actual = digest(safe_path(self.root, relative).read_bytes())
            if actual != image["sha256"]:
                raise ValueError("Image content does not match the export hash.")
            self.image_hashes.add(actual)
        self.annotations = {}
        for a in self.manifest["annotations"]:
            if a["id"] in self.annotations:
                raise ValueError("Duplicate annotation IDs.")
            if a["image_path"] not in self.images:
                raise ValueError("Missing annotation image.")
            if a["eligible_for_training"] and (
                a["label"] not in ("coma", "awake", "empty")
                or a.get("label_source") not in ("manual", "manual_occupancy")
                or a.get("occupancy") not in ("empty", "single")
            ):
                raise ValueError(
                    "Eligible targets must come from suitable manual annotations."
                )
            self.annotations[a["id"]] = a

    @lru_cache(maxsize=2)
    def image(self, path):
        value = cv2.imread(str(safe_path(self.root, path)), cv2.IMREAD_COLOR)
        if value is None:
            raise ValueError("Unreadable image.")
        return value

    @lru_cache(maxsize=64)
    def sample(self, identity):
        a = self.annotations[identity]
        crop, mask = crop_well(self.image(a["image_path"]), a["boundary_normalized"])
        features, _ = occupancy_features(crop, mask)
        return preprocess(crop, mask), features


def load_plan(dataset, path):
    plan = read_json(path)
    if (
        plan.get("kind") != "flyscope-comparison-plan"
        or plan.get("dataset_sha256") != dataset.sha
    ):
        raise ValueError(
            "Comparison plan is bound to a different dataset. Use its original export."
        )
    names = set()
    for run in plan["runs"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run["id"]) or run["id"] in names:
            raise ValueError("Invalid or duplicate run IDs.")
        names.add(run["id"])
        if not isinstance(run["seed"], int) or not 0 <= run["seed"] <= 0xFFFFFFFF:
            raise ValueError("Invalid run seed.")
        selected = run["annotation_ids"]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Empty or duplicate sample selection.")
        expected = {
            a["id"]
            for a in dataset.annotations.values()
            if a["eligible_for_training"] and a["well_uid"] in run["well_uids"]
        }
        if set(selected) != expected:
            raise ValueError(
                "Run must use exactly the eligible labels of its saved wells."
            )
    return plan


def check_holdouts(train, tests):
    for data in tests:
        if data.manifest.get("experiment_id") == train.manifest.get("experiment_id"):
            raise ValueError(
                "Training and test datasets have the same experiment/split group. Use another test grid."
            )
        if train.image_hashes & data.image_hashes:
            raise ValueError("Training and test images overlap by content hash.")


class UntrainableSelection(ValueError):
    pass


def train(
    adapter,
    dataset,
    run,
    output,
    epochs=5,
    batch_size=4,
    learning_rate=0.0001,
    mode="finetune",
):
    tf = tensorflow()
    tf.keras.utils.set_random_seed(run["seed"])
    tf.config.experimental.enable_op_determinism()
    base = adapter.load()
    model = tf.keras.models.clone_model(base)
    if mode == "finetune":
        model.set_weights(base.get_weights())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
    )
    selected = [dataset.annotations[k] for k in run["annotation_ids"]]
    states = [a for a in selected if a["label"] in CLASSES]
    if not states:
        raise UntrainableSelection(
            "No coma/awake labels in this saved selection; the binary CNN cannot be updated. Selection preserved and explicitly skipped."
        )
    # Empty annotations train occupancy only. No Empty target is sent to the
    # original two-output classifier, and all heads share the same selected wells.
    features = [dataset.sample(a["id"])[1] for a in selected]
    present = [int(a["label"] != "empty") for a in selected]
    head = fit_occupancy(features, present) or adapter.sidecar.get("occupancy_head")
    losses = []
    start = time.monotonic()
    for epoch in range(epochs):
        order = np.random.default_rng(run["seed"] + epoch).permutation(len(states))
        epoch_losses = []
        for offset in range(0, len(order), batch_size):
            batch = [states[i] for i in order[offset : offset + batch_size]]
            x = np.stack([dataset.sample(a["id"])[0] for a in batch])
            y = np.array([CLASSES[a["label"]] for a in batch])
            model.reset_metrics()
            loss = float(model.train_on_batch(x, y))
            if not np.isfinite(loss):
                raise RuntimeError(
                    "Training produced nonfinite loss; no artifact saved."
                )
            epoch_losses.append(loss)
        losses.append(float(np.mean(epoch_losses)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "model.keras"
    model.save(path, include_optimizer=False)
    metadata = {
        "kind": "flyscope-model",
        "name": f'{adapter.metadata["name"]} / {run["id"]} / {mode}',
        "artifact_sha256": digest(path.read_bytes()),
        "preprocessing": PREPROCESSING,
        "parent_version": adapter.metadata["version"],
        "classes": CLASSES,
        "occupancy_head": head,
        "dataset_sha256": dataset.sha,
        "selection": run,
        "training": {
            "mode": mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": run["seed"],
            "loss": losses,
            "seconds": time.monotonic() - start,
        },
        "test_data_used_for_training": False,
    }
    (output / "model.json").write_text(json.dumps(metadata, indent=2))
    tf.keras.backend.clear_session()
    adapter.model = None
    return KerasAdapter(path, metadata["name"], metadata)


def evaluate(adapter, dataset):
    confusion = {
        truth: {prediction: 0 for prediction in ("coma", "awake", "empty", "unknown")}
        for truth in ("coma", "awake", "empty")
    }
    predictions = []
    scored = 0
    correct = 0
    truth = {
        (a["frame_index"], a["well_uid"]): a
        for a in dataset.annotations.values()
        if a["eligible_for_training"]
    }
    for image in dataset.manifest["images"]:
        frame = image["frame_number"] - 1
        wells = []
        for w in dataset.manifest.get("wells", []):
            geometry = next(
                (
                    f
                    for f in w["frames"]
                    if f["frame_index"] == frame and f.get("active", True)
                ),
                None,
            )
            if geometry:
                wells.append(
                    {
                        "id": w["well_uid"],
                        "points": geometry["points"],
                        "partial": geometry.get("partial", False),
                    }
                )
        if not wells:
            wells = [
                {
                    "id": a["well_uid"],
                    "points": a["boundary_normalized"],
                    "partial": a.get("boundary_partial", False),
                }
                for a in dataset.annotations.values()
                if a["frame_index"] == frame
            ]
        if not wells:
            continue
        records = adapter.predict(
            safe_path(dataset.root, image["path"]).read_bytes(), wells
        )
        for record in records:
            target = truth.get((frame, record["well_id"]))
            predictions.append(
                {
                    "frame_index": frame,
                    "well_uid": record["well_id"],
                    "ground_truth": target["label"] if target else None,
                    "scored": target is not None,
                    **record,
                }
            )
            if target:
                scored += 1
                correct += record["prediction"] == target["label"]
                confusion[target["label"]][record["prediction"]] += 1
    classes = {}
    for label in confusion:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(row[label] for row in confusion.values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        classes[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if recall is not None and precision + recall
                else 0.0 if support else None
            ),
        }
    return {
        "dataset_sha256": dataset.sha,
        "experiment_id": dataset.manifest.get("experiment_id"),
        "model": adapter.metadata,
        "scored_annotations": scored,
        "unscored_predictions": len(predictions) - scored,
        "accuracy": correct / scored if scored else None,
        "confusion": confusion,
        "classes": classes,
        "predictions": predictions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--train", required=True)
    compare.add_argument("--plan")
    compare.add_argument("--test", nargs="+", required=True)
    compare.add_argument(
        "--models", nargs="+", default=["legacy-model1", "legacy-model2"]
    )
    compare.add_argument("--runs", nargs="+")
    compare.add_argument("--output", required=True)
    compare.add_argument("--epochs", type=int, default=5)
    compare.add_argument("--batch-size", type=int, default=4)
    compare.add_argument("--learning-rate", type=float, default=0.0001)
    compare.add_argument("--mode", choices=["finetune", "scratch"], default="finetune")
    scoring = sub.add_parser("evaluate")
    scoring.add_argument("--test", nargs="+", required=True)
    scoring.add_argument(
        "--models", nargs="+", default=["legacy-model1", "legacy-model2"]
    )
    scoring.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from backend.services.model_registry import ADAPTERS

    if any(
        name not in ADAPTERS or not isinstance(ADAPTERS[name], KerasAdapter)
        for name in args.models
    ):
        raise ValueError("Unknown or unsupported model trainer. Check /api/models.")
    tests = [Dataset(path, "test") for path in args.test]
    root = Path(args.output)
    if root.exists():
        raise ValueError(
            "Output directory already exists; choose a new one to preserve earlier results."
        )
    dataset = plan = None
    if args.command == "compare":
        if (
            args.epochs < 1
            or args.batch_size < 1
            or not np.isfinite(args.learning_rate)
            or args.learning_rate <= 0
        ):
            raise ValueError("Training settings must be positive.")
        dataset = Dataset(args.train, "train")
        check_holdouts(dataset, tests)
        plan = load_plan(dataset, args.plan or dataset.root / "comparison-plan.json")
        requested = set(args.runs or [r["id"] for r in plan["runs"]])
        if not requested.issubset({r["id"] for r in plan["runs"]}):
            raise ValueError("Requested run is absent from saved selections.")
    root.mkdir(parents=True)
    if plan:
        (root / "comparison-plan.json").write_text(json.dumps(plan, indent=2))
    summary = []
    for name in args.models:
        adapter = ADAPTERS[name]

        # Baselines are evaluated once. Every training run loads fresh parent
        # weights and resets the saved seed; results cannot contaminate each other.
        def score(run_name, current):
            for i, test in enumerate(tests):
                result = evaluate(current, test)
                target = root / f"{name}-{run_name}-test{i+1}.json"
                target.write_text(json.dumps(result, indent=2))
                summary.append(
                    {
                        "model_id": name,
                        "run": run_name,
                        "test_grid": test.manifest.get("experiment_id"),
                        "scored": result["scored_annotations"],
                        "accuracy": result["accuracy"],
                        "model_version": current.metadata["version"],
                        "dataset_sha256": test.sha,
                        "status": "evaluated",
                        "message": "",
                    }
                )
                print(json.dumps(summary[-1]), flush=True)

        score("baseline", adapter)
        if plan:
            for run in plan["runs"]:
                if run["id"] not in requested:
                    continue
                try:
                    current = train(
                        adapter,
                        dataset,
                        run,
                        root / name / run["id"],
                        args.epochs,
                        args.batch_size,
                        args.learning_rate,
                        args.mode,
                    )
                except UntrainableSelection as exc:
                    skipped = {
                        "model_id": name,
                        "run": run["id"],
                        "test_grid": None,
                        "scored": 0,
                        "accuracy": None,
                        "model_version": adapter.metadata["version"],
                        "dataset_sha256": None,
                        "status": "skipped",
                        "message": str(exc),
                    }
                    summary.append(skipped)
                    (root / f'{name}-{run["id"]}-skipped.json').write_text(
                        json.dumps({"selection": run, "result": skipped}, indent=2)
                    )
                    continue
                score(run["id"], current)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]) if summary else [])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
