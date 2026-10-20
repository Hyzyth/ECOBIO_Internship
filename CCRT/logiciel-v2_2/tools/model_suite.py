#!/usr/bin/env python3
"""Matched CCRT training and held-out evaluation for all three model families.

Use the app environment installed from requirements.txt. Native
SLEAP pose pretraining runs with the same Python interpreter with tools/sleap_pose.py. No test grid
is used for training, threshold selection, early stopping or normalization.
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault(
    "TORCH_HOME",
    str(
        Path(os.environ.get("FLYSCOPE_CACHE_DIR", tempfile.gettempdir()))
        / "flyscope-torch"
    ),
)
from tools.model_experiments import Dataset, load_plan, check_holdouts, safe_path
from backend.services.suite_models import (
    CLASSES,
    crop_rgb,
    torch_setup,
    SuiteModel,
    digest,
)
from backend.services.recovery_policy import apply_policy, effective_policy
from tools.training_support import CropCache, Progress
from tools.suite_validation import make_validation, validate_split, resolve_wells
from tools.suite_training import train_model


class TrainingData:
    def __init__(self, roots, progress=None):
        self.datasets = [Dataset(root, "train", progress) for root in roots]
        self.cache = None
        if len({d.sha for d in self.datasets}) != len(self.datasets):
            raise ValueError("Duplicate training datasets.")
        if len({d.manifest["experiment_id"] for d in self.datasets}) != len(
            self.datasets
        ):
            raise ValueError(
                "Training grids must have distinct experiment IDs; do not count exports of the same grid as independent individuals."
            )
        seen_images = set()
        for d in self.datasets:
            if seen_images & d.image_hashes:
                raise ValueError(
                    "Training grids share images; use one export per grid."
                )
            seen_images.update(d.image_hashes)
        self.annotations = {}
        self.occupied = {}
        self.well_names = {}
        for data in self.datasets:
            for a in data.annotations.values():
                if a["eligible_for_training"]:
                    identity = self.identity(data, a["id"])
                    self.annotations[identity] = (data, a)
                    uid = self.identity(data, a["well_uid"])
                    self.occupied[uid] = a["occupancy"] == "single"
                    self.well_names[uid] = a["well_id"]
        if not self.annotations:
            raise ValueError("No eligible manual targets.")
        self.image_maps = {
            d.sha: {
                im.get("frame_index", im["frame_number"] - 1): im
                for im in d.manifest["images"]
            }
            for d in self.datasets
        }
        self.geometry = {d.sha: d.geometry for d in self.datasets}

    @staticmethod
    def identity(data, value):
        return data.sha[:16] + "::" + value

    @lru_cache(maxsize=256)
    def rgb(self, sha, frame, well, size):
        if self.cache is not None:
            return self.cache.get(sha, frame, well, size)
        data = next(d for d in self.datasets if d.sha == sha)
        geometry = self.context_geometry(sha, frame, well)
        im = data.image(self.image_maps[sha][frame]["path"])
        return crop_rgb(im, geometry["points"], size)

    def context_geometry(self, sha, frame, well):
        if frame not in self.image_maps[sha]:
            raise ValueError(
                "Temporal context image is missing. Re-export the complete analysis window."
            )
        geometry = self.geometry[sha].get((well, frame))
        if not geometry:
            raise ValueError("Well geometry is missing from a temporal context frame.")
        if geometry.get("boundary_reviewed") is False:
            raise ValueError(
                "A temporal context boundary needs review. Approve stable boundaries or correct alignment before training."
            )
        return geometry

    def sample(self, identity, size):
        data, a = self.annotations[identity]
        return self.rgb(data.sha, a["frame_index"], a["well_uid"], size)

    def clip(self, identity, size, window):
        data, a = self.annotations[identity]
        start = data.manifest.get(
            "analysis_start_index", min(self.image_maps[data.sha])
        )
        return [
            self.rgb(
                data.sha,
                max(start, a["frame_index"] - window + 1 + i),
                a["well_uid"],
                size,
            )[0]
            for i in range(window)
        ]


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


def target(data, identity):
    return CLASSES.index(data.annotations[identity][1]["label"])


def evaluate_model(path, test, device="auto", progress=None):
    from backend.services.suite_metrics import metrics

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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--train", nargs="+", required=True)
    planning.add_argument("--output", required=True)
    planning.add_argument("--seed", type=int)
    planning.add_argument("--poor-well")
    planning.add_argument("--rich-well")
    planning.add_argument(
        "--source-plan",
        help="Existing export plan: inherit its seed and poor/rich individuals, then create a NEW validation-aware plan",
    )
    planning.add_argument("--validation-fraction", type=float, default=0.2)
    planning.add_argument("--validation-wells", nargs="+")
    planning.add_argument("--validation-unit", choices=["well", "grid"], default="well")
    planning.add_argument("--validation-folds", type=int)
    planning.add_argument("--validation-fold", type=int, default=0)
    compare = sub.add_parser("compare")
    compare.add_argument("--train", nargs="+", required=True)
    compare.add_argument("--plan")
    compare.add_argument(
        "--architectures",
        nargs="+",
        choices=["resnet18", "sleap-rf", "deepethogram"],
        default=["resnet18", "sleap-rf", "deepethogram"],
    )
    compare.add_argument("--runs", nargs="+")
    compare.add_argument("--pose-model")
    compare.add_argument("--seed", type=int, default=2026)
    compare.add_argument("--poor-well")
    compare.add_argument("--rich-well")
    compare.add_argument("--epochs", type=int, default=5)
    compare.add_argument("--flow-epochs", type=int, default=3)
    compare.add_argument("--sequence-epochs", type=int, default=5)
    compare.add_argument("--batch-size", type=int, default=4)
    compare.add_argument(
        "--resnet-batch-size", type=int, help="Override --batch-size for ResNet-18"
    )
    compare.add_argument(
        "--pose-batch-size",
        type=int,
        help="Override --batch-size for SLEAP feature extraction",
    )
    compare.add_argument(
        "--temporal-batch-size", type=int, help="Override --batch-size for DeepEthogram"
    )
    compare.add_argument(
        "--cache-dir",
        help="Reusable crop cache (default: OUTPUT_PARENT/.flyscope-training-cache)",
    )
    compare.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk crop caching; lower disk use, slower epochs",
    )
    compare.add_argument(
        "--cache-workers", type=int, default=2, help="Concurrent frame preparation jobs"
    )
    compare.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="CPU batches to prepare ahead; 0 disables prefetch",
    )
    compare.add_argument("--image-size", type=int, default=128)
    compare.add_argument("--window", type=int, default=5)
    compare.add_argument("--learning-rate", type=float, default=0.0001)
    compare.add_argument(
        "--initialization", choices=["imagenet", "scratch"], default="imagenet"
    )
    compare.add_argument("--rf-trees", type=int, default=200)
    compare.add_argument("--false-awake-cost", type=float, default=2)
    compare.add_argument(
        "--policy",
        choices=["auto", "manual"],
        default="auto",
        help="Choose recovery settings on reserved validation wells (default), or use explicit manual settings",
    )
    compare.add_argument("--confidence-threshold", type=float, default=0.8)
    compare.add_argument("--awake-frames", type=int, default=3)
    compare.add_argument(
        "--allow-initial-awake",
        action="store_true",
        help="Allow recovery confirmation without a preceding confident Coma observation (left-censored inputs)",
    )
    compare.add_argument(
        "--allow-no-validation",
        action="store_true",
        help="Explicit legacy/software-smoke opt-out; no early stopping or generalization estimate",
    )
    compare.add_argument(
        "--balance", choices=["well-class", "class", "none"], default="well-class"
    )
    compare.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Validation epochs without improvement before stopping; 0 disables early stopping",
    )
    compare.add_argument("--min-delta", type=float, default=0.0001)
    compare.add_argument("--weight-decay", type=float, default=0.0001)
    compare.add_argument("--rf-min-samples-leaf", type=int, default=5)
    compare.add_argument("--rf-max-depth", type=int, default=12)
    score = sub.add_parser("evaluate")
    score.add_argument("--models", nargs="+", required=True)
    for parser in [compare, score]:
        parser.add_argument("--test", nargs="+", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        parser.add_argument(
            "--log-every",
            type=float,
            default=5,
            help="Seconds between live progress messages; all batch losses are recorded",
        )
    args = p.parse_args(argv)

    # Dataset hashes can take minutes on a large grid. Be visible even before
    # validation succeeds and the output directory may safely be created.
    last = {}

    def announce(event, force=False, **values):
        now = time.monotonic()
        key = (event, values.get("dataset"))
        if force or now - last.get(key, -float("inf")) >= getattr(args, "log_every", 5):
            print(json.dumps({"event": event, **values}), flush=True)
            last[key] = now

    announce("starting", force=True, command=args.command, output=args.output)
    if args.command == "plan":
        data = TrainingData(args.train, announce)
        poor, rich, seed = (
            args.poor_well,
            args.rich_well,
            args.seed if args.seed is not None else 2026,
        )
        if args.source_plan:
            source = checked_plan(data, args.source_plan)
            seed = args.seed if args.seed is not None else source["seed"]
            for run in source["runs"]:
                if run["id"] == "poor-poses" and poor is None:
                    poor = run["well_uids"][0]
                if run["id"] == "rich-poses" and rich is None:
                    rich = run["well_uids"][0]
        plan = plan_for(
            data,
            seed,
            poor,
            rich,
            args.validation_fraction,
            args.validation_wells,
            args.validation_folds,
            args.validation_fold,
            args.validation_unit,
        )
        if args.source_plan:
            plan["source_plan_sha256"] = digest(args.source_plan)
        out = Path(args.output)
        if out.exists():
            raise ValueError(
                "Plan output exists; preserve the existing immutable plan."
            )
        out.write_text(json.dumps(plan, indent=2))
        return
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError(
            "Use a new output directory; results and weights are never overwritten."
        )
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    tests = [Dataset(root, "test", announce) for root in args.test]
    if args.command == "compare":
        if (
            min(
                args.epochs,
                args.flow_epochs,
                args.sequence_epochs,
                args.batch_size,
                args.rf_trees,
                args.awake_frames,
            )
            < 1
            or args.image_size < 32
            or args.image_size % 8
            or args.window < 2
            or not 0.5 <= args.confidence_threshold <= 1
            or args.false_awake_cost < 1
            or args.cache_workers < 1
            or args.prefetch_batches < 0
            or args.patience < 0
            or args.min_delta < 0
            or args.weight_decay < 0
            or args.rf_min_samples_leaf < 1
            or args.rf_max_depth < 1
            or any(
                value is not None and value < 1
                for value in (
                    args.resnet_batch_size,
                    args.pose_batch_size,
                    args.temporal_batch_size,
                )
            )
        ):
            raise ValueError("Invalid training parameters.")
        data = TrainingData(args.train, announce)
        if "deepethogram" in args.architectures and any(
            d.manifest.get("mode") != "sequence" or len(d.manifest["images"]) < 2
            for d in data.datasets
        ):
            raise ValueError(
                "DeepEthogram requires ordered multi-frame training sequences; select resnet18/sleap-rf for independent images."
            )
        for train in data.datasets:
            check_holdouts(train, tests)
        plan = checked_plan(data, args.plan, args.seed, args.poor_well, args.rich_well)
        selected = [
            r for r in plan["runs"] if args.runs is None or r["id"] in args.runs
        ]
        if args.runs and set(args.runs) - {r["id"] for r in selected}:
            raise ValueError("Requested run is not in the shared plan.")
        if not selected:
            raise ValueError("No shared runs selected.")
        args.validation = plan.get("validation")
        if not args.validation and not args.allow_no_validation:
            raise ValueError(
                "This plan has no independent validation wells. Create a NEW plan with `model_suite.py plan --source-plan ORIGINAL_PLAN --validation-fraction 0.2 ...`. Use --allow-no-validation only for an explicitly unvalidated legacy run or software smoke test."
            )
        if "sleap-rf" in args.architectures and not args.pose_model:
            raise ValueError(
                "Train the external SLEAP pose model first, then provide --pose-model."
            )
    else:
        paths = model_paths(args.models)
        data, selected, plan = None, None, None
    out.mkdir(parents=True)
    progress = Progress(out, args.log_every)
    args.progress = progress
    try:
        progress.emit(
            "validated",
            force=True,
            command=args.command,
            output=str(out),
            arguments={
                k: v
                for k, v in vars(args).items()
                if k not in ("progress", "validation")
            },
        )
        if args.command == "compare":
            (out / "comparison-plan.json").write_text(json.dumps(plan, indent=2))
            # Fail before a costly cache build if requested CUDA is unavailable.
            device = torch_setup(args.seed, args.device)
            progress.emit("device_ready", force=True, device=str(device))
            progress.emit(
                "validation_split" if args.validation else "warning_no_validation",
                force=True,
                validation={
                    k: v
                    for k, v in (args.validation or {}).items()
                    if k != "annotation_ids"
                },
            )
            progress.emit(
                "suite_start",
                force=True,
                selections=len(selected),
                architectures=args.architectures,
                training_runs=len(selected) * len(args.architectures),
            )
            if not args.no_cache:
                data.cache = CropCache.prepare(
                    data,
                    selected
                    + (
                        [{"annotation_ids": args.validation["annotation_ids"]}]
                        if args.validation
                        else []
                    ),
                    args.image_size,
                    (
                        2 * (args.window - 1)
                        if "deepethogram" in args.architectures
                        else 0
                    ),
                    args.cache_dir or out.parent / ".flyscope-training-cache",
                    progress,
                    args.cache_workers,
                )
                data.rgb.cache_clear()
            paths = []
            for kind in args.architectures:
                for run in selected:
                    paths.append(
                        train_model(kind, data, run, out / kind / run["id"], args)
                    )
        finish_evaluation(paths, tests, out, args.device, progress)
        progress.emit("complete", force=True, output=str(out), models=len(paths))
    except BaseException as error:
        progress.emit(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            force=True,
            error=str(error),
            error_type=type(error).__name__,
        )
        raise
    finally:
        if data is not None:
            data.rgb.cache_clear()
            if data.cache is not None:
                data.cache.close()
        progress.close()


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
    from backend.services.suite_metrics import write_comparison

    write_comparison(results, out)


if __name__ == "__main__":
    main()
