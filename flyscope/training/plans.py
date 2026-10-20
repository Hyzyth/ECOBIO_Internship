"""Immutable matched budgets, reserved validation and plan migration."""

import json
import random
from pathlib import Path
from flyscope.data.datasets import load_plan
from .validation import make_validation, validate_split, resolve_wells


def plan_for(
    data,
    seed=2026,
    poor=None,
    rich=None,
    validation_fraction=0,
    validation_wells=None,
    validation_folds=None,
    validation_fold=0,
    validation_unit="well",
):
    samples = data.annotations
    wells = sorted({data.identity(d, a["well_uid"]) for d, a in samples.values()})
    protected = resolve_wells(data, [value for value in (poor, rich) if value])
    validation = make_validation(
        data,
        seed,
        validation_fraction,
        protected,
        validation_wells,
        validation_folds,
        validation_fold,
        validation_unit,
    )
    if validation:
        wells = sorted(set(wells) - set(validation["well_uids"]))
    occupied = [w for w in wells if data.occupied[w]]
    runs = []
    skipped = []

    def run(name, chosen, s):
        runs.append(
            {
                "id": name,
                "seed": s,
                "well_uids": sorted(chosen),
                "annotation_ids": sorted(
                    k
                    for k, (d, a) in samples.items()
                    if data.identity(d, a["well_uid"]) in chosen
                ),
            }
        )

    run("all-annotated-wells", wells, seed)
    for count, repeats in [(25, 1), (10, 2), (5, 2), (2, 2)]:
        if len(occupied) < count:
            skipped.append(
                {
                    "wells": count,
                    "repeats": repeats,
                    "reason": "Insufficient eligible single-individual wells",
                }
            )
            continue
        for n in range(repeats):
            run_seed = (seed + count * 100003 + n) % 2**32
            run(
                f"random-{count}-{n+1}",
                random.Random(run_seed).sample(occupied, count),
                run_seed,
            )
    for name, value in [("poor-poses", poor), ("rich-poses", rich)]:
        if not value:
            continue
        matching = [w for w in occupied if w == value or data.well_names[w] == value]
        if len(matching) != 1:
            raise ValueError(
                "Poor/rich well ID is absent or ambiguous across grids; use its qualified ID from the plan."
            )
        run(name, matching, seed)
    return {
        "kind": "flyscope-suite-plan",
        "schema_version": 2 if validation else 1,
        "protocol": "full-25x1-10x2-5x2-2x2-poor-rich-v4",
        "seed": seed,
        "dataset_sha256": [d.sha for d in data.datasets],
        "runs": runs,
        "skipped": skipped,
        "selection_unit": "whole well / single individual for random subsets",
        "test_data_used": False,
        "validation": validation,
    }


def checked_plan(data, path=None, seed=2026, poor=None, rich=None):
    if path is None and len(data.datasets) == 1:
        candidate = data.datasets[0].root / "comparison-plan.json"
        if candidate.is_file():
            path = candidate
    if path:
        plan = json.loads(Path(path).read_text())
        if plan.get("kind") == "flyscope-comparison-plan" and len(data.datasets) == 1:
            original = load_plan(data.datasets[0], path)
            plan = {
                **original,
                "kind": "flyscope-suite-plan",
                "dataset_sha256": [data.datasets[0].sha],
                "runs": [
                    {
                        **r,
                        "well_uids": [
                            data.identity(data.datasets[0], w) for w in r["well_uids"]
                        ],
                        "annotation_ids": [
                            data.identity(data.datasets[0], a)
                            for a in r["annotation_ids"]
                        ],
                    }
                    for r in original["runs"]
                ],
            }
        if set(plan.get("dataset_sha256", [])) != {d.sha for d in data.datasets}:
            raise ValueError("Plan is bound to different training manifests.")
    else:
        plan = plan_for(data, seed, poor, rich)
    names = set()
    validate_split(data, plan)
    for run in plan["runs"]:
        import re

        if not re.fullmatch(r"[A-Za-z0-9_-]+", run["id"]) or run["id"] in names:
            raise ValueError("Invalid or duplicate run names.")
        names.add(run["id"])
        expected = {
            k
            for k, (d, a) in data.annotations.items()
            if data.identity(d, a["well_uid"]) in run["well_uids"]
        }
        if (
            not expected
            or set(run["annotation_ids"]) != expected
            or len(run["annotation_ids"]) != len(expected)
        ):
            raise ValueError(
                "Every model must use exactly the eligible labels of the plan's wells."
            )
    return plan
