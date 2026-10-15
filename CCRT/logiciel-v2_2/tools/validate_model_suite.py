#!/usr/bin/env python3
"""Run real training, matched subsets, holdout scoring and weight reload on CPU.

This validates software, not biological accuracy. Supply the separately trained
SLEAP pose smoke model. All generated files remain outside the repository.
"""
import argparse
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
assert len(selections["all-annotated-wells"]["annotation_ids"]) == 18
assert len(selections["poor-poses"]["annotation_ids"]) == 6
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
print(
    "PASS actual ResNet, SLEAP/RF and DeepEthogram flow/two-stream/TGM training; identical full/poor/rich selections, partial labels, independent evaluation and all nine checkpoint reloads:",
    out,
)
