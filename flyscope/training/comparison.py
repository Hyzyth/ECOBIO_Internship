#!/usr/bin/env python3
"""Matched CCRT training and held-out evaluation for all three model families.

Use the app environment installed from requirements.txt. Native
SLEAP pose pretraining runs with the same Python interpreter with tools/sleap_pose.py. No test grid
is used for training, threshold selection, early stopping or normalization.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault(
    "TORCH_HOME",
    str(
        Path(os.environ.get("FLYSCOPE_CACHE_DIR", tempfile.gettempdir()))
        / "flyscope-torch"
    ),
)
from flyscope.data.datasets import Dataset, check_holdouts
from flyscope.models.runtime import torch_setup, digest
from flyscope.training.support import CropCache, Progress
from flyscope.training.trainers import train_model


from .data import TrainingData
from .plans import plan_for, checked_plan
from flyscope.evaluation import model_paths, finish_evaluation


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
    compare.add_argument(
        "--backbone",
        choices=["finetune", "frozen"],
        default="finetune",
        help="Controlled low-data ablation: frozen keeps ImageNet feature CNNs fixed; only heads/motion/temporal modules train",
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
        if args.backbone == "frozen" and args.initialization != "imagenet":
            raise ValueError(
                "A frozen backbone requires ImageNet initialization; do not freeze random features."
            )
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


if __name__ == "__main__":
    main()
