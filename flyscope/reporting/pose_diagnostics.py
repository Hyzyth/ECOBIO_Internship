#!/usr/bin/env python3
"""Inspect pose quality on anatomical labels or on actual CCRT wells.

The two modes answer different questions. Native SLP labels support localization
metrics. CCRT state annotations support overlays, not anatomical accuracy scores.
"""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np
import torch
from flyscope.models.runtime import torch_setup, crop_rgb, pose_input
from flyscope.training.pose import (
    read_meta,
    network,
    centered_examples,
    infer_images,
    clear_cache,
    sha,
)
from flyscope.training.pose_quality import evaluate, write_overlays


def diagnose(args):
    root, output = Path(args.pose_model).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError("Pose diagnostic output must be new.")
    meta = read_meta(root)
    config = meta["configuration"]
    device = torch_setup(device=args.device)
    output.mkdir(parents=True)
    if args.labels:
        nodes, examples, _ = centered_examples(args.labels, config["size"])
        if nodes != config["nodes"]:
            raise ValueError("Anatomical node names/order differ from the model.")
        # Default to the stored held-out identities when evaluating its source SLP.
        split = (meta.get("validation") or {}).get("split") or {}
        identities = split.get("validation_examples")
        if identities and not args.all_labels:
            if sha(args.labels) != meta["source_labels_sha256"]:
                raise ValueError(
                    "These are different anatomical labels; use --all-labels explicitly for an external evaluation dataset."
                )
            # Some native SLP files contain multiple labeled-frame records for
            # the same video/frame, with reused local instance indices.
            precise = all("label_record" in e for e in identities)

            def identity(e):
                return (
                    e.get("video_sha256") or Path(e["video"]).name,
                    e["frame"],
                    e["instance"],
                    e.get("label_record") if precise else None,
                )

            expected = {identity(e) for e in identities}
            examples = [e for e in examples if identity(e[3]) in expected]
            if not examples:
                raise ValueError(
                    "The source labels do not contain the saved validation identities. For a different annotated dataset use --all-labels explicitly."
                )
            scope = (
                "saved validation identities; source video/frame provenance must match"
            )
        else:
            scope = "all supplied labels; these may have been used for training"
        model = network(config).to(device)
        model.load_state_dict(
            torch.load(root / "pose.pt", map_location=device, weights_only=True)
        )
        metrics, rows = evaluate(
            model, examples, config, device, args.batch_size, sigma=meta.get("sigma", 3)
        )
        metrics["evaluation_scope"] = scope
        metrics["labels_sha256_match_training"] = (
            sha(args.labels) == meta["source_labels_sha256"]
        )
    else:
        from flyscope.data.datasets import Dataset

        manifest = json.loads((Path(args.dataset) / "annotations.json").read_text())
        data = Dataset(args.dataset, manifest.get("role", "train"))
        by_state = defaultdict(list)
        for a in data.annotations.values():
            if (
                args.wells
                and a["well_id"] not in args.wells
                and a["well_uid"] not in args.wells
            ):
                continue
            if a.get("eligible_for_training"):
                by_state[a["label"]].append(a)
        chosen = []
        per_state = max(1, args.max_examples // max(1, len(by_state)))
        for state, annotations in sorted(by_state.items()):
            annotations.sort(key=lambda a: (a["well_id"], a["frame_index"]))
            chosen.extend(
                annotations[i]
                for i in np.linspace(
                    0, len(annotations) - 1, min(len(annotations), per_state), dtype=int
                )
            )
        if not chosen:
            raise ValueError(
                "No eligible manually annotated examples match the selected wells."
            )
        examples = []
        for a in chosen:
            rgb, _ = crop_rgb(
                data.image(a["image_path"]), a["boundary_normalized"], config["size"]
            )
            prepared = pose_input(rgb)
            examples.append(
                (
                    prepared,
                    np.full((len(config["nodes"]), 2), np.nan),
                    np.zeros(len(config["nodes"]), bool),
                    {
                        "caption": f"{a['well_id']} / frame {a['frame_number']} / {a['label']}",
                        "frame": a["frame_index"],
                        "well_id": a["well_id"],
                        "state": a["label"],
                    },
                )
            )
        try:
            poses = infer_images(
                root,
                [e[0] for e in examples],
                list(map(str, range(len(examples)))),
                args.device,
                batch_size=args.batch_size,
            )
        finally:
            clear_cache()
        rows = [
            {
                "identity": e[3],
                "predicted_xy": [
                    [p["x"] * config["size"], p["y"] * config["size"]]
                    for p in pose["points"]
                ],
                "scores": [p["score"] for p in pose["points"]],
                "ground_truth_xy": [[-1, -1]] * len(config["nodes"]),
                "visible": [False] * len(config["nodes"]),
            }
            for e, pose in zip(examples, poses)
        ]
        metrics = {
            "evaluation_scope": "qualitative CCRT-domain preview",
            "examples": len(rows),
            "state_examples": {
                state: sum(e[3]["state"] == state for e in examples)
                for state in by_state
            },
            "note": "Coma/Awake/Empty labels are not keypoint coordinates. No anatomical accuracy/PCK can be inferred from this preview; peak scores are not calibrated confidence in correctness.",
        }
    write_overlays(examples, rows, config["nodes"], output, args.max_examples)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "predictions.json").write_text(json.dumps(rows, indent=2))
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Pose quality</title><style>body{font:16px system-ui;margin:32px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Pose quality</h1><p>Magenta: predicted peaks; gray: peaks below 0.05; green crosses: human anatomical labels, when available. Numbers follow nodes.json. Inspect Coma and Awake separately, especially folded/occluded legs.</p><pre>'
        + html.escape(json.dumps(metrics, indent=2))
        + '</pre><img src="overlays.jpg" alt="Pose predictions and optional human keypoints">'
    )
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-model", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--labels", help="Native SLP with anatomical keypoints")
    inputs.add_argument(
        "--dataset", help="Extracted CCRT export; qualitative state-stratified preview"
    )
    parser.add_argument("--wells", nargs="+")
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="Evaluate all supplied anatomical labels, including potentially seen examples",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.max_examples < 1 or args.batch_size < 1:
        parser.error("Example and batch counts must be positive.")
    diagnose(args)


if __name__ == "__main__":
    main()
