#!/usr/bin/env python3
"""Run real training, matched subsets, holdout scoring and weight reload on CPU.

This validates software, not biological accuracy. Supply the separately trained
SLEAP pose smoke model. All generated files remain outside the repository.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.suite_fixtures import grids
from tools.model_suite import main, TrainingData, plan_for

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--pose-model", required=True)
p.add_argument("--output")
args = p.parse_args()
out = Path(args.output or tempfile.mkdtemp(prefix="flyscope-suite-"))
out.mkdir(parents=True, exist_ok=True)
train, test = grids(out / "fixtures")
plan = plan_for(TrainingData([train]), 2026, "W1", "W2")
(out / "plan.json").write_text(json.dumps(plan, indent=2))
main(
    [
        "compare",
        "--train",
        str(train),
        "--test",
        str(test),
        "--plan",
        str(out / "plan.json"),
        "--output",
        str(out / "results"),
        "--runs",
        "all-annotated-wells",
        "poor-poses",
        "rich-poses",
        "--pose-model",
        args.pose_model,
        "--epochs",
        "1",
        "--flow-epochs",
        "1",
        "--sequence-epochs",
        "1",
        "--rf-trees",
        "5",
        "--batch-size",
        "4",
        "--resnet-batch-size",
        "2",
        "--pose-batch-size",
        "2",
        "--temporal-batch-size",
        "2",
        "--image-size",
        "64",
        "--window",
        "3",
        "--initialization",
        "scratch",
        "--device",
        "cpu",
    ]
)
manifests = list((out / "results").glob("*/*/model.json"))
assert len(manifests) == 9
selections = {}
for path in manifests:
    m = json.loads(path.read_text())
    run = m["selection"]["id"]
    if run in selections:
        assert selections[run] == m["selection"]
    selections[run] = m["selection"]
    assert not m["training"]["test_data_used_for_training"]
    assert m["training"]["batch_size"] == 2
assert len(selections["all-annotated-wells"]["annotation_ids"]) == 18
assert len(selections["poor-poses"]["annotation_ids"]) == 6
with (out / "results/training.csv").open() as stream:
    training_rows = list(csv.DictReader(stream))
epoch_rows = [row for row in training_rows if row["event"] == "epoch"]
assert len(epoch_rows) == 12  # 3 appearance + 3 * (flow, two-stream, TGM)
assert len([row for row in training_rows if row["event"] == "batch"]) == 60
assert (out / "results/loss.png").stat().st_size > 1000
for path in manifests:
    model = json.loads(path.read_text())
    for phase, history in model["training"]["loss"].items():
        logged = [
            float(row["loss"])
            for row in epoch_rows
            if row["architecture"] == model["architecture"]
            and row["run"] == model["selection"]["id"]
            and row["phase"] == phase
        ]
        assert logged == history
events = [
    json.loads(row) for row in (out / "results/progress.jsonl").read_text().splitlines()
]
assert events[-1]["event"] == "complete"
assert len([row for row in events if row["event"] == "model_saved"]) == 9
summary = json.loads((out / "results/summary.json").read_text())
assert len(summary) == 9
assert all(
    s["scored_annotations"] == 18 and s["unscored_predictions"] == 18 for s in summary
)
main(
    [
        "evaluate",
        "--models",
        str(out / "results"),
        "--test",
        str(test),
        "--output",
        str(out / "reloaded"),
        "--device",
        "cpu",
    ]
)
# Changing input delivery must not change learned targets, shuffle order or
# pixel values. Verify the optimized path against the uncached CPU path.
main(
    [
        "compare",
        "--train",
        str(train),
        "--test",
        str(test),
        "--plan",
        str(out / "plan.json"),
        "--architectures",
        "resnet18",
        "--runs",
        "poor-poses",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--image-size",
        "64",
        "--initialization",
        "scratch",
        "--device",
        "cpu",
        "--no-cache",
        "--prefetch-batches",
        "0",
        "--output",
        str(out / "uncached"),
    ]
)
import torch

cached = torch.load(
    out / "results/resnet18/poor-poses/model.pt", map_location="cpu", weights_only=True
)
uncached = torch.load(
    out / "uncached/resnet18/poor-poses/model.pt", map_location="cpu", weights_only=True
)
assert all(torch.equal(cached[key], uncached[key]) for key in cached)
print(
    "PASS actual ResNet, SLEAP/RF and DeepEthogram flow/two-stream/TGM training; live losses/plots, batch overrides, identical cached/uncached weights, full/poor/rich selections, partial labels, independent evaluation and all nine checkpoint reloads:",
    out,
)
