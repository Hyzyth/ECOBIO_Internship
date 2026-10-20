#!/usr/bin/env python3
"""Create a compact, shareable audit without copying images, weights or predictions.

Run on an existing comparison directory (old exports work). Full per-frame JSON
is read one model at a time. The resulting ZIP is suitable for debugging a large
experiment without re-uploading gigabytes of repeated frame records.
"""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np


def model_audit(result):
    model, metrics = result["model"], result["metrics"]
    records = result.get("predictions", [])
    first = {}
    for r in records:
        well = r["well_id"]
        if well not in first or r["frame_index"] < first[well]["frame_index"]:
            first[well] = r
    false_awake_scores = [
        r.get("raw_confidence", r.get("confidence"))
        for r in records
        if r.get("ground_truth") == "coma" and r.get("raw_prediction") == "awake"
    ]
    false_awake_scores = [s for s in false_awake_scores if s is not None]
    classes = metrics["raw"]["classes"]
    support = {k: v["support"] for k, v in classes.items()}
    total = sum(support.values())
    compact = {
        "architecture": model["architecture"],
        "run": model["selection"]["id"],
        "version": model["version"],
        "test_grid": result["test_experiment"],
        "training_class_support": model["class_support"],
        "test_class_support": support,
        "always_awake_accuracy": support.get("awake", 0) / total if total else None,
        "majority_baseline_accuracy": max(support.values()) / total if total else None,
        "raw_accuracy": metrics["raw"]["accuracy"],
        "decision_accuracy": metrics["overall"]["accuracy"],
        "raw_balanced_accuracy": metrics["raw"]["balanced_accuracy"],
        "raw_coma_recall": classes["coma"]["recall"],
        "raw_awake_precision": classes["awake"]["precision"],
        "raw_awake_recall": classes["awake"]["recall"],
        "first_frame_raw_awake_wells": sum(
            r.get("raw_prediction") == "awake" for r in first.values()
        ),
        "first_frame_decision_awake_wells": sum(
            r.get("prediction") == "awake" for r in first.values()
        ),
        "wells": len(first),
        "false_awake_confidence_quantiles": (
            {
                str(q): float(np.quantile(false_awake_scores, q))
                for q in (0, 0.25, 0.5, 0.75, 1)
            }
            if false_awake_scores
            else {}
        ),
        "training": model.get("training"),
        "validation": model.get("validation"),
        "recovery_policy": model["recovery_policy"],
        # Metrics contain censoring and paired denominators, not source images.
        "metrics": metrics,
    }
    return compact


def audit(root, output, pose=None, training_log=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(
            "Audit output must be new; existing results are never overwritten."
        )
    output.mkdir(parents=True)
    reports = []
    for path in sorted(root.rglob("*.json")):
        if (
            not path.name.startswith(("resnet18-", "deepethogram-", "sleap-rf-"))
            or output in path.parents
        ):
            continue
        with path.open() as stream:
            result = json.load(stream)
        if "model" in result and "metrics" in result:
            reports.append(model_audit(result))
            print(
                f"Audited {reports[-1]['architecture']} / {reports[-1]['run']}",
                flush=True,
            )
    if not reports:
        raise ValueError(
            "No per-model evaluation JSON found; point --results at the complete comparison output."
        )
    (output / "model-audits.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False)
    )
    columns = [
        k
        for k in reports[0]
        if k
        not in (
            "metrics",
            "training",
            "validation",
            "recovery_policy",
            "false_awake_confidence_quantiles",
        )
    ]
    with (output / "audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in reports)
    for source in [Path(training_log)] if training_log else root.rglob("training.csv"):
        if output in source.parents:
            continue
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [r for r in reader if r.get("event") == "epoch"]
        if not rows:
            continue
        with (output / "training-epochs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        break
    if pose:
        meta = json.loads(Path(pose).read_text())
        zero = np.pi * meta.get("sigma", 3) ** 2 / meta["configuration"]["size"] ** 2
        (output / "pose-training.json").write_text(
            json.dumps(
                {
                    "configuration": meta["configuration"],
                    "epochs": meta["epochs"],
                    "loss": meta["loss"],
                    "examples": len(meta.get("examples", [])),
                    "annotated_frames": len(
                        {(e["video"], e["frame"]) for e in meta.get("examples", [])}
                    ),
                    "approximate_zero_heatmap_mse": (
                        float(zero) if not meta.get("loss_method") else None
                    ),
                    "note": "Continuous Gaussian approximation for the legacy full-image MSE only; not a localization score or proof of collapse.",
                },
                indent=2,
            )
        )
    from flyscope.reporting.audit_plots import write_audit_plots

    write_audit_plots(reports, output)
    archive = output.with_suffix(".zip")
    if archive.exists():
        raise ValueError("Audit ZIP already exists; choose a new output name.")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file():
                z.write(path, path.name)
    print(f"Compact audit: {archive}", flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pose-json")
    parser.add_argument(
        "--training-log",
        help="Optional training.csv outside the result directory; only epoch rows are included",
    )
    args = parser.parse_args()
    audit(args.results, args.output, args.pose_json, args.training_log)


if __name__ == "__main__":
    main()
