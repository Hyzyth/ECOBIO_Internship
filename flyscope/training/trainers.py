"""Offline trainers: fit on selected wells, select checkpoints on shared validation wells.

The CLI and immutable data selections live in model_suite.py. Runtime model
architectures and inference remain in flyscope/models/runtime.py.
"""

from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np
from flyscope.models.runtime import (
    CLASSES,
    PREPROCESSING,
    POSE_PREPROCESSING,
    PoseWorker,
    make_resnet,
    make_deepethogram,
    torch_setup,
    normalize,
    frozen_batch_norm,
    flow_loss,
    pose_features,
    forest_export,
    digest,
    pose_input,
    model_version,
    POSE_QUALITY_POLICY,
    usable_pose,
)
from flyscope.training.policy_selection import select_policy
from flyscope.training.support import prefetched, host_tensor, device_tensor
from flyscope.training.validation import (
    sample_weights,
    validation_metrics,
    BestCheckpoint,
)


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
    validation = args.validation
    validation_ids = validation["annotation_ids"] if validation else []
    fit_weights = sample_weights(data, ids, args.balance, args.false_awake_cost)
    val_weights = (
        sample_weights(data, validation_ids, "well-class", 1) if validation_ids else {}
    )
    labels = [target(data, k) for k in ids]
    start = time.monotonic()
    losses = {}
    validation_losses = {}
    selected_epochs = {}
    validation_result = None
    fit_result = None
    validation_probabilities = None
    validation_eligible = None
    progress.emit(
        "label_support",
        force=True,
        **context,
        training=dict(Counter(CLASSES[i] for i in labels)),
        validation=validation.get("class_support") if validation else None,
        balancing=args.balance,
        false_awake_cost=args.false_awake_cost,
    )
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
        pose_eligible = []
        extraction_ids = ids + validation_ids
        try:
            batches = (
                extraction_ids[offset : offset + batch_size]
                for offset in range(0, len(extraction_ids), batch_size)
            )

            def prepare_pose(batch):
                crops = [data.sample(k, args.image_size) for k in batch]
                return crops, [pose_input(c[0]) for c in crops]

            progress.emit(
                "phase_start",
                force=True,
                **context,
                phase="pose_features",
                samples=len(extraction_ids),
            )
            completed = 0
            for batch, (crops, images) in prefetched(
                batches, prepare_pose, args.prefetch_batches
            ):
                poses = worker.predict(images, batch, batch_size=batch_size)
                features.extend(pose_features(p, c[1]) for p, c in zip(poses, crops))
                pose_eligible.extend(usable_pose(p) for p in poses)
                completed += len(batch)
                progress.emit(
                    "pose_features",
                    **context,
                    phase="pose_features",
                    samples=completed,
                    total_samples=len(extraction_ids),
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
            min_samples_leaf=args.rf_min_samples_leaf,
            max_depth=args.rf_max_depth,
        )
        state_indices = [i for i, y in enumerate(labels) if y != 2]
        if state_indices:
            forest.fit(
                np.stack(features)[state_indices],
                np.asarray(labels)[state_indices],
                sample_weight=[fit_weights[ids[i]] for i in state_indices],
            )
        occupancy = RandomForestClassifier(
            n_estimators=args.rf_trees,
            random_state=run["seed"],
            n_jobs=2,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_depth=args.rf_max_depth,
        )
        occupancy.fit(
            np.stack([data.sample(k, args.image_size)[1] for k in ids]),
            [2 if y == 2 else 0 for y in labels],
            sample_weight=list(
                sample_weights(
                    data, ids, args.balance, args.false_awake_cost, occupancy=True
                ).values()
            ),
        )
        payload = {
            "state": forest_export(forest) if state_indices else None,
            "occupancy": forest_export(occupancy),
            "state_features": "normalized keypoint distances, visibility and confidence only",
            "occupancy_features": "relative foreground components only",
        }

        def probabilities_for(model, x):
            result = np.zeros((len(x), 3))
            result[:, model.classes_.astype(int)] = model.predict_proba(x)
            return result

        occupied_fit = probabilities_for(
            occupancy, np.stack([data.sample(k, args.image_size)[1] for k in ids])
        )
        states_fit = (
            probabilities_for(forest, np.stack(features[: len(ids)]))
            if state_indices
            else np.zeros((len(ids), 3))
        )
        fit_probabilities = states_fit * occupied_fit[:, 0, None]
        fit_probabilities[:, 2] = occupied_fit[:, 2]
        fit_result = validation_metrics(
            fit_probabilities, labels, [fit_weights[k] for k in ids]
        )
        if validation_ids:
            occupied = probabilities_for(
                occupancy,
                np.stack([data.sample(k, args.image_size)[1] for k in validation_ids]),
            )
            states = (
                probabilities_for(forest, np.stack(features[len(ids) :]))
                if state_indices
                else np.zeros((len(validation_ids), 3))
            )
            probabilities = states * occupied[:, 0, None]
            probabilities[:, 2] = occupied[:, 2]
            validation_probabilities = probabilities
            validation_eligible = [
                ok or int(p.argmax()) == 2
                for ok, p in zip(pose_eligible[len(ids) :], probabilities)
            ]
            validation_result = validation_metrics(
                probabilities,
                [target(data, k) for k in validation_ids],
                [val_weights[k] for k in validation_ids],
            )
            progress.emit(
                "validation",
                force=True,
                **context,
                phase="random_forests",
                **validation_result,
            )
        progress.fit_metrics(
            **context,
            phase="random_forests",
            loss=fit_result["balanced_nll"],
            validation_loss=(validation_result or {}).get("balanced_nll"),
            training_balanced_accuracy=fit_result["balanced_accuracy"],
            training_coma_recall=fit_result["class_recall"].get("coma"),
            validation_balanced_accuracy=(validation_result or {}).get(
                "balanced_accuracy"
            ),
            validation_coma_recall=(validation_result or {})
            .get("class_recall", {})
            .get("coma"),
        )
        progress.plot()
    else:
        network = (
            make_resnet(args.initialization)
            if kind == "resnet18"
            else make_deepethogram(args.window, args.initialization)
        )
        if args.backbone == "frozen":
            if kind == "resnet18":
                for name, parameter in network.named_parameters():
                    parameter.requires_grad = name.startswith("fc.")
            else:
                for backbone in (network.spatial, network.flow):
                    for parameter in backbone.parameters():
                        parameter.requires_grad = False
        network.to(device)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        def classification_loss(logits, batch):
            weights = torch.tensor(
                [fit_weights[k] if k in fit_weights else val_weights[k] for k in batch],
                device=device,
                dtype=torch.float32,
            )
            # Dataset-normalized weights keep minority/cost weighting effective
            # even when a mini-batch contains only one class (or one sample).
            return (criterion(logits, targets(batch)) * weights).mean()

        def optimize(phase, parameters, epochs, batch_loss, prepare, predict=None):
            optimizer = torch.optim.AdamW(
                parameters, lr=args.learning_rate, weight_decay=args.weight_decay
            )
            history = []
            val_history = []
            best = BestCheckpoint(args.patience, args.min_delta)
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
                validation_loss = None
                validation_stats = {}
                if validation_ids:
                    network.eval()
                    progress.emit(
                        "validation_start",
                        force=True,
                        **context,
                        phase=phase,
                        epoch=epoch + 1,
                        samples=len(validation_ids),
                    )
                    batches = (
                        validation_ids[offset : offset + batch_size]
                        for offset in range(0, len(validation_ids), batch_size)
                    )
                    val_losses, probabilities = [], []
                    completed = 0
                    with torch.no_grad():
                        for batch, inputs in prefetched(
                            batches, prepare, args.prefetch_batches
                        ):
                            if predict is not None:
                                logits = predict(batch, inputs)
                                value = classification_loss(logits, batch)
                                probabilities.extend(logits.softmax(1).cpu().numpy())
                            else:
                                value = batch_loss(batch, inputs)
                            val_losses.append(float(value.detach()) * len(batch))
                            completed += len(batch)
                            progress.emit(
                                "validating",
                                **context,
                                phase=phase,
                                epoch=epoch + 1,
                                samples=completed,
                                total_samples=len(validation_ids),
                            )
                    validation_loss = float(sum(val_losses) / len(validation_ids))
                    val_history.append(validation_loss)
                    if predict is not None:
                        stats = validation_metrics(
                            probabilities,
                            [target(data, k) for k in validation_ids],
                            [val_weights[k] for k in validation_ids],
                        )
                        validation_stats = {
                            "validation_balanced_accuracy": stats["balanced_accuracy"],
                            "validation_coma_recall": stats["class_recall"].get("coma"),
                        }
                    stop = best.update(validation_loss, epoch + 1, network)
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
                    validation_loss=validation_loss,
                    **validation_stats,
                )
                progress.plot()
                if validation_ids and stop:
                    progress.emit(
                        "early_stopping",
                        force=True,
                        **context,
                        phase=phase,
                        epoch=epoch + 1,
                        selected_epoch=best.epoch,
                        validation_loss=best.best,
                    )
                    break
            losses[phase] = history
            validation_losses[phase] = val_history
            if validation_ids:
                best.restore(network)
                selected_epochs[phase] = best.epoch
                progress.emit(
                    "checkpoint_selected",
                    force=True,
                    **context,
                    phase=phase,
                    selected_epoch=best.epoch,
                    validation_loss=best.best,
                )

            if validation_ids and predict is not None:
                # Search recovery settings using the restored best checkpoint,
                # not the final epoch or any test predictions.
                network.eval()
                probabilities = []
                batches = (
                    validation_ids[offset : offset + batch_size]
                    for offset in range(0, len(validation_ids), batch_size)
                )
                with torch.no_grad():
                    for batch, inputs in prefetched(
                        batches, prepare, args.prefetch_batches
                    ):
                        probabilities.extend(
                            predict(batch, inputs).softmax(1).cpu().numpy()
                        )
                return np.asarray(probabilities)
            return None

        def targets(batch):
            return torch.tensor([target(data, k) for k in batch], device=device)

        if kind == "resnet18":
            validation_probabilities = optimize(
                "appearance",
                [p for p in network.parameters() if p.requires_grad],
                args.epochs,
                lambda batch, inputs: classification_loss(
                    network(normalize(device_tensor(inputs, device))), batch
                ),
                lambda batch: host_tensor(
                    [data.sample(k, args.image_size)[0] for k in batch],
                    device.type == "cuda",
                ),
                predict=lambda batch, inputs: network(
                    normalize(device_tensor(inputs, device))
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
                    if p.requires_grad and not n.startswith(("motion.", "temporal."))
                ],
                args.epochs,
                lambda batch, inputs: classification_loss(
                    network.features(clips(batch, inputs))[1], batch
                ),
                prepare_clips,
                predict=lambda batch, inputs: network.features(clips(batch, inputs))[1],
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

            keys = sorted(
                {key for k in ids + validation_ids for key in history_keys(k)}
            )
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

                    validation_probabilities = optimize(
                        "temporal_tgm",
                        network.temporal.parameters(),
                        args.sequence_epochs,
                        lambda batch, inputs: classification_loss(
                            network.temporal(
                                inputs.to(device, non_blocking=device.type == "cuda")
                            )[:, :, -1],
                            batch,
                        ),
                        prepare_temporal,
                        predict=lambda batch, inputs: network.temporal(
                            inputs.to(device, non_blocking=device.type == "cuda")
                        )[:, :, -1],
                    )
                finally:
                    features._mmap.close()
        payload = network.state_dict()
    policy = {
        "confidence_threshold": args.confidence_threshold,
        "awake_consecutive_frames": args.awake_frames,
        "require_coma_before_awake": not args.allow_initial_awake,
        "automatic_confirmation": True,
        "source": "manual",
        "infer_matching_gaps": True,
    }
    candidates = []
    if args.policy == "auto":
        policy, candidates = select_policy(
            data,
            validation_ids,
            (
                validation_probabilities
                if validation_probabilities is not None
                else np.empty((0, 3))
            ),
            args.confidence_threshold,
            args.awake_frames,
            not args.allow_initial_awake,
            eligible=validation_eligible,
        )
    progress.emit("policy_selected", force=True, **context, policy=policy)
    out.mkdir(parents=True, exist_ok=False)
    (out / "recovery-policy-search.json").write_text(
        json.dumps({"policy": policy, "candidates": candidates}, indent=2)
    )
    if kind == "sleap-rf":
        (out / "forest.json").write_text(json.dumps(payload))
        (out / "pose").mkdir()
        for name in ("pose.pt", "pose.json"):
            shutil.copy2(pose_source / name, out / "pose" / name)
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
        "backbone_mode": args.backbone if kind != "sleap-rf" else None,
        "pose_parent_sha256": (
            digest(pose_source / "pose.pt") if pose_source else None
        ),
        "recovery_policy": policy,
        "validation": (
            {
                **{k: v for k, v in validation.items() if k != "annotation_ids"},
                "selected_epochs": selected_epochs,
                "forest_metrics": validation_result,
                "checkpoint_selection": "lowest held-out balanced NLL (photometric loss for optical flow)",
            }
            if validation
            else None
        ),
        "training_warnings": [
            f"No {label} training examples; this state is unsupported by these data."
            for label in CLASSES
            if label not in Counter(CLASSES[i] for i in labels)
        ],
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
            "fit_metrics": fit_result,
            "validation_loss": validation_losses,
            "balancing": args.balance,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "selected_epochs": selected_epochs,
            "rf_min_samples_leaf": args.rf_min_samples_leaf,
            "rf_max_depth": args.rf_max_depth,
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
        from flyscope.models.upstream import WHEEL_SHA256

        meta["upstream_architectures_sha256"] = WHEEL_SHA256
    if kind == "sleap-rf":
        meta["pose_quality_policy"] = dict(POSE_QUALITY_POLICY)
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
