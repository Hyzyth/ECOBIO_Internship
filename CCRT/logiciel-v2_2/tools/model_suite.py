#!/usr/bin/env python3
"""Matched CCRT training and held-out evaluation for all three model families.

Use the app environment installed from requirements.txt. Native
SLEAP pose pretraining runs with the same Python interpreter with tools/sleap_pose.py. No test grid
is used for training, threshold selection, early stopping or normalization.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
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
    PREPROCESSING,
    crop_rgb,
    make_resnet,
    make_deepethogram,
    torch_setup,
    tensor,
    normalize,
    frozen_batch_norm,
    flow_loss,
    PoseWorker,
    pose_features,
    forest_export,
    SuiteModel,
    digest,
    pose_input,
    POSE_PREPROCESSING,
    model_version,
)
import numpy as np
from tools.training_support import (
    CropCache,
    Progress,
    prefetched,
    host_tensor,
    device_tensor,
)


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


def plan_for(data, seed=2026, poor=None, rich=None):
    samples = data.annotations
    wells = sorted({data.identity(d, a["well_uid"]) for d, a in samples.values()})
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
    for count, repeats in [(25, 1), (10, 2), (5, 4), (2, 5)]:
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
        "schema_version": 1,
        "protocol": "full-25x1-10x2-5x4-2x5-poor-rich-v3",
        "seed": seed,
        "dataset_sha256": [d.sha for d in data.datasets],
        "runs": runs,
        "skipped": skipped,
        "selection_unit": "whole well / single individual for random subsets",
        "test_data_used": False,
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


def train_model(kind, data, run, out, args):
    import torch

    device = torch_setup(run["seed"], args.device)
    progress = args.progress
    batch_size = (
        getattr(
            args,
            {
                "resnet18": "resnet_batch_size",
                "sleap-rf": "pose_batch_size",
                "deepethogram": "temporal_batch_size",
            }[kind],
        )
        or args.batch_size
    )
    context = {"architecture": kind, "run": run["id"]}
    progress.emit(
        "model_start",
        force=True,
        **context,
        device=str(device),
        samples=len(run["annotation_ids"]),
        batch_size=batch_size,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )
    ids = run["annotation_ids"]
    labels = [target(data, k) for k in ids]
    start = time.monotonic()
    losses = {}
    pose_source = None
    if kind == "sleap-rf":
        if not args.pose_model:
            raise ValueError(
                "SLEAP/RF needs a trained pose model. Run sleap_pose.py train on the creator's SLP dataset first."
            )
        from sklearn.ensemble import RandomForestClassifier

        pose_source = Path(args.pose_model).resolve()
        worker = PoseWorker(pose_source, device=args.device)
        features = []
        try:
            batches = (
                ids[offset : offset + batch_size]
                for offset in range(0, len(ids), batch_size)
            )

            def prepare_pose(batch):
                crops = [data.sample(k, args.image_size) for k in batch]
                return crops, [pose_input(c[0]) for c in crops]

            progress.emit(
                "phase_start",
                force=True,
                **context,
                phase="pose_features",
                samples=len(ids),
            )
            completed = 0
            for batch, (crops, images) in prefetched(
                batches, prepare_pose, args.prefetch_batches
            ):
                poses = worker.predict(images, batch)
                features.extend(pose_features(p, c[1]) for p, c in zip(poses, crops))
                completed += len(batch)
                progress.emit(
                    "pose_features",
                    **context,
                    phase="pose_features",
                    samples=completed,
                    total_samples=len(ids),
                )
        finally:
            worker.close()
        progress.emit(
            "phase_start",
            force=True,
            **context,
            phase="random_forests",
            device="cpu",
            trees=args.rf_trees,
        )
        forest = RandomForestClassifier(
            n_estimators=args.rf_trees,
            random_state=run["seed"],
            n_jobs=2,
            class_weight={0: args.false_awake_cost, 1: 1, 2: args.false_awake_cost},
        )
        state_indices = [i for i, y in enumerate(labels) if y != 2]
        if state_indices:
            forest.fit(
                np.stack(features)[state_indices], np.asarray(labels)[state_indices]
            )
        occupancy = RandomForestClassifier(
            n_estimators=args.rf_trees,
            random_state=run["seed"],
            n_jobs=2,
            class_weight={0: 1, 2: args.false_awake_cost},
        )
        occupancy.fit(
            np.stack([data.sample(k, args.image_size)[1] for k in ids]),
            [2 if y == 2 else 0 for y in labels],
        )
        payload = {
            "state": forest_export(forest) if state_indices else None,
            "occupancy": forest_export(occupancy),
            "state_features": "normalized keypoint distances, visibility and confidence only",
            "occupancy_features": "relative foreground components only",
        }
    else:
        network = (
            make_resnet(args.initialization)
            if kind == "resnet18"
            else make_deepethogram(args.window, args.initialization)
        )
        network.to(device)
        weights = torch.tensor(
            [args.false_awake_cost, 1, args.false_awake_cost],
            device=device,
            dtype=torch.float32,
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        def optimize(phase, parameters, epochs, batch_loss, prepare):
            optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
            history = []
            progress.emit(
                "phase_start",
                force=True,
                **context,
                phase=phase,
                epochs=epochs,
                batch_size=batch_size,
            )
            for epoch in range(epochs):
                order = np.random.default_rng(run["seed"] + epoch).permutation(len(ids))
                values = []
                network.train()
                frozen_batch_norm(network)
                batches = (
                    [ids[i] for i in order[offset : offset + batch_size]]
                    for offset in range(0, len(order), batch_size)
                )
                iterator = iter(prefetched(batches, prepare, args.prefetch_batches))
                epoch_start = time.monotonic()
                input_wait = 0
                completed = 0
                total_batches = (len(ids) + batch_size - 1) // batch_size
                progress.emit(
                    "epoch_start",
                    force=True,
                    **context,
                    phase=phase,
                    epoch=epoch + 1,
                    epochs=epochs,
                    batches=total_batches,
                )
                try:
                    for index in range(total_batches):
                        waiting = time.monotonic()
                        batch, inputs = next(iterator)
                        input_wait += time.monotonic() - waiting
                        optimizer.zero_grad(set_to_none=True)
                        loss = batch_loss(batch, inputs)
                        scalar = float(loss.detach())
                        if not np.isfinite(scalar):
                            raise RuntimeError(
                                "Nonfinite training loss; no artifact saved."
                            )
                        loss.backward()
                        optimizer.step()
                        values.append(scalar)
                        completed += len(batch)
                        elapsed = time.monotonic() - epoch_start
                        gpu = (
                            {
                                "gpu_allocated_mib": torch.cuda.memory_allocated(device)
                                / 1024**2,
                                "gpu_reserved_mib": torch.cuda.memory_reserved(device)
                                / 1024**2,
                            }
                            if device.type == "cuda"
                            else {}
                        )
                        progress.batch(
                            **context,
                            phase=phase,
                            epoch=epoch + 1,
                            epochs=epochs,
                            batch=index + 1,
                            batches=total_batches,
                            samples=completed,
                            total_samples=len(ids),
                            loss=scalar,
                            elapsed_s=elapsed,
                            samples_per_s=completed / max(elapsed, 1e-9),
                            input_wait_s=input_wait,
                            eta_s=(len(ids) - completed) * elapsed / completed,
                            **gpu,
                        )
                finally:
                    iterator.close()
                history.append(float(np.mean(values)))
                progress.epoch(
                    **context,
                    phase=phase,
                    epoch=epoch + 1,
                    epochs=epochs,
                    loss=history[-1],
                    elapsed_s=time.monotonic() - epoch_start,
                    samples=len(ids),
                    samples_per_s=len(ids) / max(time.monotonic() - epoch_start, 1e-9),
                    input_wait_s=input_wait,
                )
                progress.plot()
            losses[phase] = history

        def targets(batch):
            return torch.tensor([target(data, k) for k in batch], device=device)

        if kind == "resnet18":
            optimize(
                "appearance",
                network.parameters(),
                args.epochs,
                lambda batch, inputs: criterion(
                    network(normalize(device_tensor(inputs, device))),
                    targets(batch),
                ),
                lambda batch: host_tensor(
                    [data.sample(k, args.image_size)[0] for k in batch],
                    device.type == "cuda",
                ),
            )
        else:
            if any(
                d.manifest.get("mode") != "sequence" or len(d.manifest["images"]) < 2
                for d in data.datasets
            ):
                raise ValueError(
                    "DeepEthogram training requires ordered multi-frame sequences."
                )

            def prepare_clips(batch):
                return host_tensor(
                    [
                        im
                        for k in batch
                        for im in data.clip(k, args.image_size, args.window)
                    ],
                    device.type == "cuda",
                )

            def clips(batch, inputs):
                return device_tensor(inputs, device).reshape(
                    len(batch), args.window, 3, args.image_size, args.image_size
                )

            optimize(
                "optical_flow",
                network.motion.parameters(),
                args.flow_epochs,
                lambda batch, inputs: flow_loss(network, clips(batch, inputs)),
                prepare_clips,
            )
            network.motion.eval()
            for p in network.motion.parameters():
                p.requires_grad = False
            optimize(
                "two_stream",
                [
                    p
                    for n, p in network.named_parameters()
                    if not n.startswith(("motion.", "temporal."))
                ],
                args.epochs,
                lambda batch, inputs: criterion(
                    network.features(clips(batch, inputs))[1], targets(batch)
                ),
                prepare_clips,
            )
            network.eval()

            def history_keys(identity):
                d, a = data.annotations[identity]
                start_frame = d.manifest.get(
                    "analysis_start_index", min(data.image_maps[d.sha])
                )
                return [
                    (
                        d.sha,
                        max(start_frame, a["frame_index"] - args.window + 1 + i),
                        a["well_uid"],
                    )
                    for i in range(args.window)
                ]

            keys = sorted({key for k in ids for key in history_keys(k)})
            indices = {key: i for i, key in enumerate(keys)}
            starts = {
                d.sha: d.manifest.get(
                    "analysis_start_index", min(data.image_maps[d.sha])
                )
                for d in data.datasets
            }

            def prepare_features(batch):
                return host_tensor(
                    [
                        data.rgb(
                            sha,
                            max(starts[sha], frame - args.window + 1 + i),
                            well,
                            args.image_size,
                        )[0]
                        for sha, frame, well in batch
                        for i in range(args.window)
                    ],
                    device.type == "cuda",
                )

            # Feature extraction was one GPU call per frame/well, with every
            # feature retained on the GPU. Batch it and keep the fixed features
            # on disk instead; TGM training now has bounded device memory.
            progress.emit(
                "phase_start",
                force=True,
                **context,
                phase="temporal_features",
                samples=len(keys),
                batch_size=batch_size,
            )
            with tempfile.TemporaryDirectory(
                prefix="temporal-features-", dir=args.output
            ) as temporary:
                if (
                    shutil.disk_usage(temporary).free
                    < len(keys) * 1024 * 4 + 16 * 1024**2
                ):
                    raise ValueError(
                        "Not enough free space in --output for temporary temporal features."
                    )
                features = np.lib.format.open_memmap(
                    Path(temporary) / "features.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(keys), 1024),
                )
                try:
                    batches = (
                        keys[offset : offset + batch_size]
                        for offset in range(0, len(keys), batch_size)
                    )
                    completed = 0
                    with torch.no_grad():
                        for batch, inputs in prefetched(
                            batches, prepare_features, args.prefetch_batches
                        ):
                            values = (
                                network.features(clips(batch, inputs))[0].cpu().numpy()
                            )
                            features[completed : completed + len(batch)] = values
                            completed += len(batch)
                            progress.emit(
                                "temporal_features",
                                **context,
                                phase="temporal_features",
                                samples=completed,
                                total_samples=len(keys),
                            )
                    features.flush()

                    def prepare_temporal(batch):
                        rows = [indices[key] for k in batch for key in history_keys(k)]
                        value = torch.from_numpy(
                            features[rows]
                            .reshape(len(batch), args.window, 1024)
                            .transpose(0, 2, 1)
                            .copy()
                        )
                        return value.pin_memory() if device.type == "cuda" else value

                    optimize(
                        "temporal_tgm",
                        network.temporal.parameters(),
                        args.sequence_epochs,
                        lambda batch, inputs: criterion(
                            network.temporal(
                                inputs.to(device, non_blocking=device.type == "cuda")
                            )[:, :, -1],
                            targets(batch),
                        ),
                        prepare_temporal,
                    )
                finally:
                    features._mmap.close()
        payload = network.state_dict()
    out.mkdir(parents=True, exist_ok=False)
    if kind == "sleap-rf":
        (out / "forest.json").write_text(json.dumps(payload))
        shutil.copytree(pose_source, out / "pose")
    else:
        torch.save(payload, out / "model.pt")
    artifacts = {
        str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()
    }
    meta = {
        "kind": "flyscope-suite-model",
        "architecture": kind,
        "name": f"{kind} / {run['id']}",
        "image_size": args.image_size,
        "window": args.window if kind == "deepethogram" else 1,
        "requires_sequence": kind == "deepethogram",
        "preprocessing": PREPROCESSING,
        "classes": CLASSES,
        "pose_preprocessing": POSE_PREPROCESSING if kind == "sleap-rf" else None,
        "selection": run,
        "class_support": dict(Counter(CLASSES[i] for i in labels)),
        "training_dataset_sha256": [d.sha for d in data.datasets],
        "training_experiment_ids": [d.manifest["experiment_id"] for d in data.datasets],
        "training_image_sha256": sorted(
            set().union(*(d.image_hashes for d in data.datasets))
        ),
        "artifact_hashes": artifacts,
        "initialization": (
            args.initialization
            if kind != "sleap-rf"
            else "fixed external SLEAP pose model"
        ),
        "pose_parent_sha256": (
            digest(pose_source / "pose.pt") if pose_source else None
        ),
        "recovery_policy": {
            "confidence_threshold": args.confidence_threshold,
            "awake_consecutive_frames": args.awake_frames,
        },
        "training": {
            "epochs": args.epochs,
            "flow_epochs": args.flow_epochs,
            "sequence_epochs": args.sequence_epochs,
            "rf_trees": args.rf_trees,
            "seed": run["seed"],
            "learning_rate": args.learning_rate,
            "batch_size": batch_size,
            "false_awake_cost": args.false_awake_cost,
            "loss": losses,
            "seconds": time.monotonic() - start,
            "test_data_used_for_training": False,
        },
    }
    meta["runtime"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "determinism": (
            "strict"
            if device.type == "cpu"
            else "seeded; unsupported deterministic kernels warn"
        ),
    }
    if kind == "deepethogram":
        from backend.services.upstream_architectures import WHEEL_SHA256

        meta["upstream_architectures_sha256"] = WHEEL_SHA256
    meta["version"] = model_version(meta)
    (out / "model.json").write_text(json.dumps(meta, indent=2))
    progress.emit(
        "model_saved",
        force=True,
        **context,
        path=str(out / "model.json"),
        seconds=time.monotonic() - start,
    )
    return out / "model.json"


def apply_policy(records, policy):
    threshold = policy["confidence_threshold"]
    count = policy["awake_consecutive_frames"]
    streak = 0
    confirmed = None
    onset = None
    previous = None
    for index, r in enumerate(records):
        if previous is not None and r["frame_index"] != previous + 1:
            streak = 0
        previous = r["frame_index"]
        raw = r.get("raw_prediction", r["prediction"])
        confidence = r["confidence"]
        r["raw_prediction"] = raw
        r.setdefault("raw_confidence", confidence)
        if confirmed is not None:
            r["prediction"] = "awake"
            r["confidence"] = None
        elif confidence is None or confidence < threshold:
            r["prediction"] = "unsure"
            streak = 0
        elif raw == "awake":
            streak += 1
            if streak >= count:
                confirmed = r["frame_index"]
                onset = records[index - streak + 1]["frame_index"]
                for candidate in records[index - streak + 1 : index + 1]:
                    candidate["prediction"] = "awake"
                    candidate["recovery_confirmed_frame_index"] = confirmed
                    candidate["recovery_onset_frame_index"] = onset
            else:
                r["prediction"] = "unsure"
        else:
            streak = 0
        r["recovery_confirmed_frame_index"] = confirmed
        r["recovery_onset_frame_index"] = onset


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
                    [r for r in output if r["well_id"] == uid], meta["recovery_policy"]
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
    planning.add_argument("--seed", type=int, default=2026)
    planning.add_argument("--poor-well")
    planning.add_argument("--rich-well")
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
    compare.add_argument("--confidence-threshold", type=float, default=0.8)
    compare.add_argument("--awake-frames", type=int, default=3)
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
        plan = plan_for(data, args.seed, args.poor_well, args.rich_well)
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
            arguments=vars(args) | {"progress": None},
        )
        if args.command == "compare":
            (out / "comparison-plan.json").write_text(json.dumps(plan, indent=2))
            # Fail before a costly cache build if requested CUDA is unavailable.
            device = torch_setup(args.seed, args.device)
            progress.emit("device_ready", force=True, device=str(device))
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
                    selected,
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
