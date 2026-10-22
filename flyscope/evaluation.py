"""Frozen-checkpoint evaluation; no weight or policy fitting on test grids."""

import json
from pathlib import Path
from flyscope.data.datasets import safe_path
from flyscope.models.runtime import SuiteModel
from flyscope.models.recovery import apply_policy, effective_policy


def evaluate_model(path, test, device="auto", progress=None):
    from flyscope.reporting.metrics import metrics

    if progress:
        progress.emit(
            "evaluation_loading",
            force=True,
            phase="evaluation",
            model=str(path),
            test=test.manifest["experiment_id"],
        )
    model = SuiteModel(path, device)
    meta = model.meta
    if meta["architecture"] == "deepethogram" and (
        test.manifest.get("mode") != "sequence" or len(test.manifest["images"]) < 2
    ):
        model.close()
        raise ValueError("DeepEthogram evaluation requires an ordered image sequence.")
    if (
        test.sha in meta["training_dataset_sha256"]
        or test.manifest["experiment_id"] in meta["training_experiment_ids"]
        or test.image_hashes & set(meta["training_image_sha256"])
    ):
        model.close()
        raise ValueError(
            "Test grid overlaps the model's training data or experiment group."
        )
    truth = {
        (a["frame_index"], a["well_uid"]): a
        for a in test.annotations.values()
        if a["eligible_for_training"]
    }
    geometry_by_frame = test.geometry
    output = []
    context = {
        "architecture": meta["architecture"],
        "run": meta["selection"]["id"],
        "phase": "evaluation",
        "test": test.manifest["experiment_id"],
    }
    if progress:
        progress.emit(
            "evaluation_start",
            force=True,
            **context,
            total_images=len(test.manifest["images"]),
        )
    try:
        for index, image in enumerate(
            sorted(test.manifest["images"], key=lambda im: im["frame_number"]), 1
        ):
            frame = image.get("frame_index", image["frame_number"] - 1)
            wells = []
            for w in test.manifest["wells"]:
                g = geometry_by_frame.get((w["well_uid"], frame))
                if g:
                    wells.append(
                        {
                            "id": w["well_uid"],
                            "points": g["points"],
                            "partial": g.get("partial", False),
                        }
                    )
            if not wells:
                continue
            records = model.predict(
                safe_path(test.root, image["path"]).read_bytes(),
                wells,
                session=test.sha,
                frame=frame,
            )
            if progress:
                progress.emit(
                    "evaluating",
                    force=index == len(test.manifest["images"]),
                    **context,
                    images=index,
                    total_images=len(test.manifest["images"]),
                )
            for r in records:
                a = truth.get((frame, r["well_id"]))
                w = next(
                    w for w in test.manifest["wells"] if w["well_uid"] == r["well_id"]
                )
                output.append(
                    {
                        **r,
                        "frame_index": frame,
                        "frame_id": image.get("frame_id") or image["original_filename"],
                        "filename": image["original_filename"],
                        "ground_truth": a["label"] if a else None,
                        "scored": a is not None,
                        "species": w.get("species", test.manifest.get("species")),
                        "well_name": w["well_id"],
                    }
                )
        if test.manifest.get("mode") == "sequence":
            for uid in {r["well_id"] for r in output}:
                apply_policy(
                    [r for r in output if r["well_id"] == uid],
                    effective_policy(
                        meta["recovery_policy"], test.manifest.get("interval")
                    ),
                )
        else:
            for r in output:
                r.setdefault("raw_prediction", r["prediction"])
                r.setdefault("raw_confidence", r["confidence"])
                if (
                    r["confidence"] is None
                    or r["confidence"] < meta["recovery_policy"]["confidence_threshold"]
                ):
                    r["prediction"] = "unsure"
        result = {
            "model": meta,
            "applied_recovery_policy": effective_policy(
                meta["recovery_policy"], test.manifest.get("interval")
            ),
            "test_dataset_sha256": test.sha,
            "test_timing": {
                k: test.manifest.get(k)
                for k in ("mode", "interval", "time_origin", "experiment_id")
            },
            "test_experiment": test.manifest["experiment_id"],
            "metrics": metrics(output, test.manifest),
            "predictions": output,
        }
        return result
    finally:
        model.close()


def model_paths(values):
    paths = []
    for value in values:
        p = Path(value).resolve()
        candidates = [p] if p.is_file() else sorted(p.rglob("model.json"))
        paths.extend(
            c
            for c in candidates
            if json.loads(c.read_text()).get("kind") == "flyscope-suite-model"
        )
    if not paths:
        raise ValueError("No trained suite model manifests found.")
    return list(dict.fromkeys(paths))


def finish_evaluation(paths, tests, out, device, progress):
    results = []
    for path in paths:
        for test in tests:
            result = evaluate_model(path, test, device, progress)
            results.append(result)
            name = f"{result['model']['architecture']}-{result['model']['version'][:12]}-{test.sha[:12]}.json"
            (out / name).write_text(json.dumps(result, indent=2, allow_nan=False))
            print(
                json.dumps(
                    {
                        "model": result["model"]["name"],
                        "test": test.manifest["experiment_id"],
                        "metrics": result["metrics"]["overall"],
                    }
                ),
                flush=True,
            )
    from flyscope.reporting.metrics import write_comparison

    write_comparison(results, out)
