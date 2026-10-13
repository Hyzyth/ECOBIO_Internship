"""Held-out metrics distinguish raw classification, abstention and recovery."""

import csv
import html
import json
from pathlib import Path
import numpy as np

LABELS = ["coma", "awake", "empty"]


def classification(rows, field="prediction"):
    scored = [r for r in rows if r.get("scored") and r.get("ground_truth") in LABELS]
    confusion = {
        label: {p: 0 for p in LABELS + ["unsure", "unknown"]} for label in LABELS
    }
    for r in scored:
        confusion[r["ground_truth"]][r[field]] += 1
    classes = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(v[label] for v in confusion.values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        classes[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if recall and precision
                else 0.0 if support else None
            ),
        }
    supported = [c for c in classes.values() if c["support"]]
    n = len(scored)
    decided = [r for r in scored if r[field] in LABELS]
    correct = sum(r[field] == r["ground_truth"] for r in scored)
    return {
        "scored_annotations": n,
        "unscored_predictions": len(rows) - n,
        "accuracy": correct / n if n else None,
        "coverage": len(decided) / n if n else None,
        "accuracy_when_decided": correct / len(decided) if decided else None,
        "balanced_accuracy": (
            float(np.mean([c["recall"] for c in supported])) if supported else None
        ),
        "macro_f1": float(np.mean([c["f1"] for c in supported])) if supported else None,
        "false_awake": sum(confusion[k]["awake"] for k in ["coma", "empty"]),
        "confusion": confusion,
        "classes": classes,
    }


def first_recovery(rows, truth=False):
    # Original indices, never a newly renumbered trimmed sequence.
    ordered = sorted(rows, key=lambda r: r["frame_index"])
    states = [r.get("ground_truth") if truth else r["prediction"] for r in ordered]
    awake = next((i for i, s in enumerate(states) if s == "awake"), None)
    if awake is None:
        complete = all(s in LABELS for s in states)
        status = "not_observed" if complete else "incompletely_annotated"
        return {"status": status, "frame_index": None, "frame_id": None}
    row = ordered[awake]
    if awake == 0:
        status = "already_awake_left_censored"
    elif (
        states[awake - 1] != "coma"
        or ordered[awake - 1]["frame_index"] != row["frame_index"] - 1
    ):
        status = "uncertain_interval"
    elif "coma" in states[awake + 1 :]:
        status = "contradictory_later_coma"
    else:
        status = "observed"
    return {
        "status": status,
        "frame_index": row["frame_index"],
        "frame_id": row["frame_id"],
        "filename": row["filename"],
    }


def metrics(rows, manifest):
    overall = classification(rows)
    raw = classification(rows, "raw_prediction")
    scored = [r for r in rows if r.get("scored")]
    calibration = {"brier": None, "ece": None, "bins": [], "auc": {}}
    if scored:
        probabilities = np.array(
            [[r["state_probabilities"].get(k, 0) for k in LABELS] for r in scored]
        )
        targets = np.array([LABELS.index(r["ground_truth"]) for r in scored])
        onehot = np.eye(3)[targets]
        calibration["brier"] = float(
            np.mean(np.sum((probabilities - onehot) ** 2, axis=1))
        )
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == targets
        ece = 0.0
        for i in range(10):
            chosen = (confidence >= i / 10) & (
                confidence < ((i + 1) / 10) if i < 9 else confidence <= 1
            )
            if not chosen.any():
                continue
            count = int(chosen.sum())
            accuracy = float(correct[chosen].mean())
            score = float(confidence[chosen].mean())
            ece += count / len(scored) * abs(accuracy - score)
            calibration["bins"].append(
                {
                    "low": i / 10,
                    "high": (i + 1) / 10,
                    "count": count,
                    "raw_accuracy": accuracy,
                    "confidence": score,
                }
            )
        calibration["ece"] = ece
        from sklearn.metrics import roc_auc_score, average_precision_score

        for i, label in enumerate(LABELS):
            binary = targets == i
            calibration["auc"][label] = {
                "roc_auc": (
                    float(roc_auc_score(binary, probabilities[:, i]))
                    if binary.any() and not binary.all()
                    else None
                ),
                "average_precision": (
                    float(average_precision_score(binary, probabilities[:, i]))
                    if binary.any()
                    else None
                ),
            }
    species = {
        name: classification(
            [r for r in rows if (r.get("species") or "not specified") == name]
        )
        for name in sorted({r.get("species") or "not specified" for r in rows})
    }
    recovery = []
    for uid in (
        sorted({r["well_id"] for r in rows})
        if manifest.get("mode") == "sequence"
        else []
    ):
        well = [r for r in rows if r["well_id"] == uid]
        if not any(r["scored"] for r in well):
            continue
        truth = first_recovery(well, True)
        predicted = first_recovery(well)
        error = None
        if truth["status"] == "observed" and predicted["status"] == "observed":
            error = predicted["frame_index"] - truth["frame_index"]
        origin = manifest.get("time_origin") or {}
        interval = manifest.get("interval")

        def seconds(event):
            return (
                origin.get("offset_seconds", 0) + event["frame_index"] * interval
                if origin.get("confirmed")
                and interval
                and event["status"] == "observed"
                else None
            )

        recovery.append(
            {
                "well_uid": uid,
                "well_id": well[0]["well_name"],
                "species": well[0].get("species"),
                "ground_truth": truth,
                "predicted": predicted,
                "error_frames": error,
                "ground_truth_ccrt_seconds": seconds(truth),
                "predicted_ccrt_seconds": seconds(predicted),
                "missed_recovery": truth["status"] == "observed"
                and predicted["frame_index"] is None,
                "false_recovery": predicted["frame_index"] is not None
                and truth["status"] == "not_observed",
            }
        )
    errors = [r["error_frames"] for r in recovery if r["error_frames"] is not None]
    recovery_summary = {
        "paired_observed_wells": len(errors),
        "mae_frames": float(np.mean(np.abs(errors))) if errors else None,
        "bias_frames": float(np.mean(errors)) if errors else None,
        "missed_recoveries": sum(r["missed_recovery"] for r in recovery),
        "false_recoveries": sum(r["false_recovery"] for r in recovery),
        "censored_predicted_recoveries": sum(
            r["predicted"]["status"] == "already_awake_left_censored" for r in recovery
        ),
        "uncertain_predicted_recoveries": sum(
            r["predicted"]["status"] == "uncertain_interval" for r in recovery
        ),
        "wells": recovery,
    }
    return {
        "overall": {
            k: v for k, v in overall.items() if k not in ["confusion", "classes"]
        },
        "decision_classes": overall["classes"],
        "decision_confusion": overall["confusion"],
        "raw": raw,
        "calibration": calibration,
        "species": species,
        "recovery": recovery_summary,
        "definitions": [
            "Only eligible human annotations are scored. Unannotated wells are never negatives.",
            "Decision accuracy includes abstentions as incorrect; coverage and accuracy_when_decided expose that tradeoff.",
            "Raw metrics score original frame probabilities before confidence/streak/irreversibility rules.",
            "Recovery frame errors use paired observed transitions; censored or uncertain truth is excluded.",
            "Confidence thresholds and streak lengths are fixed before held-out evaluation; no test calibration or tuning occurs.",
        ],
    }


def write_comparison(results, out):
    rows = []
    for r in results:
        m = r["model"]
        metrics = r["metrics"]
        rows.append(
            {
                "model": m["architecture"],
                "run": m["selection"]["id"],
                "version": m["version"],
                "test_grid": r["test_experiment"],
                "selected_wells": len(m["selection"]["well_uids"]),
                "selected_annotations": len(m["selection"]["annotation_ids"]),
                "confidence_threshold": m["recovery_policy"]["confidence_threshold"],
                "awake_consecutive_frames": m["recovery_policy"][
                    "awake_consecutive_frames"
                ],
                **metrics["overall"],
                "raw_accuracy": metrics["raw"]["accuracy"],
                "raw_macro_f1": metrics["raw"]["macro_f1"],
                "ece": metrics["calibration"]["ece"],
                "brier": metrics["calibration"]["brier"],
                "recovery_mae_frames": metrics["recovery"]["mae_frames"],
                "missed_recoveries": metrics["recovery"]["missed_recoveries"],
                "false_recoveries": metrics["recovery"]["false_recoveries"],
            }
        )
    (out / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    with (out / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        "'" + v
                        if isinstance(v, str)
                        and v.lstrip().startswith(("=", "+", "-", "@"))
                        else v
                    )
                    for k, v in row.items()
                }
            )
    columns = [
        "model",
        "run",
        "test_grid",
        "selected_wells",
        "accuracy",
        "macro_f1",
        "coverage",
        "accuracy_when_decided",
        "raw_macro_f1",
        "recovery_mae_frames",
        "false_awake",
    ]
    table = (
        "<table><thead><tr>"
        + "".join("<th>" + html.escape(c.replace("_", " ")) + "</th>" for c in columns)
        + "</tr></thead><tbody>"
    )
    for row in rows:
        table += (
            "<tr>"
            + "".join(
                "<td>"
                + html.escape(
                    "—"
                    if row[c] is None
                    else (
                        str(round(row[c], 4))
                        if isinstance(row[c], float)
                        else str(row[c])
                    )
                )
                + "</td>"
                for c in columns
            )
            + "</tr>"
        )
    table += "</tbody></table>"
    document = (
        """<!doctype html><html><head><meta charset="utf-8"><title>CCRT model comparison</title><style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccc;text-align:left}th{cursor:pointer;background:#edf3f7}p{max-width:900px}</style></head><body><h1>Held-out CCRT model comparison</h1><p>Every architecture uses the same saved well selections. Accuracy includes Unsure decisions; coverage and accuracy when decided show abstention. Raw F1 measures appearance/pose/temporal classification before the irreversible recovery policy. Recovery MAE excludes censored or uncertain ground truth. Click a column to sort.</p>"""
        + table
        + """<p>See per-grid JSON for confusion matrices, class/species metrics, calibration, original probabilities, frame IDs and recovery censoring. summary.csv contains the full comparison table.</p><script>document.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{const b=document.querySelector('tbody'),r=[...b.rows];r.sort((a,c)=>{const x=a.cells[i].textContent,y=c.cells[i].textContent;return !isNaN(+x)&&!isNaN(+y)?+y-+x:x.localeCompare(y)});r.forEach(t=>b.append(t))})</script></body></html>"""
    )
    (out / "comparison.html").write_text(document)
