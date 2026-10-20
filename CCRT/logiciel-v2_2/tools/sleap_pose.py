#!/usr/bin/env python3
"""SLEAP-NN PyTorch anatomical pose training and inference in the app environment.

Native SLP keypoints provide anatomical targets, never Coma/Awake labels.
"""
import argparse
import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

MODEL_CACHE = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def network(config):
    import torch
    from sleap_nn.architectures.unet import UNet

    class PoseNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = UNet(
                in_channels=3,
                output_stride=1,
                filters=config["filters"],
                filters_rate=2,
                down_blocks=3,
                up_blocks=3,
            )
            self.head = torch.nn.Conv2d(config["filters"], len(config["nodes"]), 1)

        def forward(self, images):
            return self.head(self.backbone(images)["outputs"][-1])

    return PoseNetwork()


def read_meta(root):
    root = Path(root)
    meta = json.loads((root / "pose.json").read_text())
    if meta.get("framework") != "pytorch":
        raise ValueError(
            "This is a legacy TensorFlow pose checkpoint. Retrain the pose estimator from its original .slp labels with tools/sleap_pose.py train, then retrain the SLEAP/RF selections. Image/annotation exports remain compatible."
        )
    if sha(root / "pose.pt") != meta["weights_sha256"]:
        raise ValueError("Pose weights do not match their metadata hash.")
    return meta


def clear_cache():
    MODEL_CACHE.clear()


def centered_examples(labels, size):
    import sleap_io as sio

    source = Path(labels).resolve()
    data = sio.load_slp(str(source), open_videos=False)
    for video in data.videos:
        filename = str(video.filename)
        candidates = [
            Path(filename),
            source.parent / filename,
            source.parent / filename.replace("\\", "/").split("/")[-1],
        ]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise ValueError(
                f"SLP video is missing: {filename}. Keep its video next to the labels."
            )
        video.replace_filename(str(found.resolve()))
    if len(data.skeletons) != 1:
        raise ValueError("Use one anatomical skeleton per pose model.")
    nodes = data.skeletons[0].node_names
    examples = []
    image_cache = {}
    video_hashes = {str(v.filename): sha(v.filename) for v in data.videos}
    for label_record, frame in enumerate(data.labeled_frames):
        if not frame.user_instances:
            continue
        key = (frame.video.filename, frame.frame_idx)
        if key not in image_cache:
            image_cache[key] = frame.video[frame.frame_idx]
        image = image_cache[key]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        for n, instance in enumerate(frame.user_instances):
            points = instance.numpy()
            visible = np.isfinite(points).all(axis=1)
            if visible.sum() < 2:
                continue
            low, high = points[visible].min(axis=0), points[visible].max(axis=0)
            extent = max(float((high - low).max()) * 1.5, 16)
            center = (low + high) / 2
            left, top = np.floor(center - extent / 2).astype(int)
            # White padding and a square instance crop preserve anatomy at edges.
            width = int(np.ceil(extent))
            crop = np.full((width, width, 3), 255, np.uint8)
            x0, y0 = max(0, left), max(0, top)
            x1, y1 = min(image.shape[1], left + width), min(image.shape[0], top + width)
            if x1 <= x0 or y1 <= y0:
                continue
            crop[y0 - top : y1 - top, x0 - left : x1 - left] = image[y0:y1, x0:x1, :3]
            xy = (points - [left, top]) * size / width
            examples.append(
                (
                    cv2.resize(crop, (size, size)),
                    xy,
                    visible,
                    {
                        "video": str(frame.video.filename),
                        "video_sha256": video_hashes[str(frame.video.filename)],
                        "frame": frame.frame_idx,
                        "instance": n,
                        "label_record": label_record,
                    },
                )
            )
    if not examples:
        raise ValueError(
            "Native labels contain no usable user keypoints; predicted instances are not targets."
        )
    return nodes, examples, data


def train(args):
    import torch
    from importlib.metadata import version
    from backend.services.suite_models import torch_setup, tensor
    from tools.pose_quality import (
        split_examples,
        heatmap_targets,
        heatmap_loss,
        augment_example,
        evaluate,
        write_overlays,
    )
    from tools.suite_validation import BestCheckpoint
    from tools.training_support import Progress

    device = torch_setup(args.seed, args.device)

    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError("Pose output must be a new directory.")
    parent = None
    if args.parent:
        parent = read_meta(args.parent)
        args.size = parent["configuration"]["size"]
        args.filters = parent["configuration"]["filters"]
    nodes, examples, data = centered_examples(args.labels, args.size)
    if args.max_examples:
        order = np.random.default_rng(args.seed).permutation(len(examples))[
            : args.max_examples
        ]
        examples = [examples[i] for i in order]
    validation_examples = []
    validation_split = None
    if not args.allow_no_validation:
        examples, validation_examples, validation_split = split_examples(
            examples, args.validation_fraction, args.seed, args.validation_gap_frames
        )
        if parent:
            parent_hashes = {
                v["path"]: v["sha256"] for v in parent.get("source_videos", [])
            }

            def identity(record):
                content = record.get("video_sha256") or parent_hashes.get(
                    record["video"]
                )
                if content is None:
                    matches = {
                        v["sha256"]
                        for v in parent.get("source_videos", [])
                        if v["path"].replace("\\", "/").split("/")[-1]
                        == record["video"].replace("\\", "/").split("/")[-1]
                    }
                    content = (
                        next(iter(matches)) if len(matches) == 1 else record["video"]
                    )
                return content, record["frame"]

            seen = {identity(e) for e in parent.get("examples", [])}
            if any(identity(e[3]) in seen for e in validation_examples):
                raise ValueError(
                    "The parent pose model already trained on proposed validation frames. Use untouched anatomical labels or retain its original grouped split; otherwise this is not independent validation."
                )
    config = {"size": args.size, "filters": args.filters, "nodes": nodes}
    model = network(config).to(device)
    if parent:
        if (
            parent["configuration"]["nodes"] != nodes
            or sha(Path(args.parent) / "pose.pt") != parent["weights_sha256"]
        ):
            raise ValueError(
                "Parent pose skeleton or weights hash differs from the provided labels."
            )
        model.load_state_dict(
            torch.load(
                Path(args.parent) / "pose.pt", map_location=device, weights_only=True
            )
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses, validation_history = [], []
    best = BestCheckpoint(args.patience, min_delta=1e-8)
    out.mkdir(parents=True)
    progress = Progress(out)
    try:
        progress.emit(
            "pose_split",
            force=True,
            training_examples=len(examples),
            validation_examples=len(validation_examples),
            split_unit=(validation_split or {}).get("unit"),
        )
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = []
            order = np.random.default_rng(args.seed + epoch).permutation(len(examples))
            for offset in range(0, len(order), args.batch_size):
                batch = [examples[i] for i in order[offset : offset + args.batch_size]]
                if not args.no_augmentation:
                    batch = [
                        augment_example(
                            e,
                            np.random.default_rng(
                                args.seed + epoch * len(examples) + int(i)
                            ),
                        )
                        for e, i in zip(batch, order[offset : offset + args.batch_size])
                    ]
                x = tensor([e[0] for e in batch], device)
                target, mask = heatmap_targets(batch, args.size, args.sigma, device)
                optimizer.zero_grad(set_to_none=True)
                loss = heatmap_loss(model(x), target, mask)
                value = float(loss.detach().cpu())
                if not np.isfinite(value):
                    raise RuntimeError("Nonfinite pose loss; no artifact saved.")
                loss.backward()
                optimizer.step()
                epoch_loss.append(value)
                progress.batch(
                    architecture="sleap-pose",
                    run="anatomical-labels",
                    phase="pose",
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=offset // args.batch_size + 1,
                    batches=(len(order) + args.batch_size - 1) // args.batch_size,
                    loss=value,
                    samples=min(offset + args.batch_size, len(order)),
                    total_samples=len(order),
                )
            losses.append(float(np.mean(epoch_loss)))
            values = {}
            stop = False
            if validation_examples:
                metrics, _ = evaluate(
                    model,
                    validation_examples,
                    config,
                    device,
                    args.batch_size,
                    sigma=args.sigma,
                )
                validation_history.append(metrics)
                values = {
                    "validation_loss": metrics["balanced_heatmap_loss"],
                    "validation_pck_005": metrics["pck_005"],
                    "validation_mean_error_px": metrics["mean_error_px"],
                    "validation_peak_coverage": metrics["visible_prediction_coverage"],
                }
                selection_score = (1 - metrics["pck_005"]) * metrics[
                    "visible_targets"
                ] + min(0.499999, metrics["mean_error_px"] / args.size / 4)
                stop = best.update(selection_score, epoch + 1, model)
            progress.epoch(
                architecture="sleap-pose",
                run="anatomical-labels",
                phase="pose",
                epoch=epoch + 1,
                epochs=args.epochs,
                loss=losses[-1],
                **values,
            )
            progress.plot()
            if stop:
                progress.emit(
                    "early_stopping",
                    force=True,
                    selected_epoch=best.epoch,
                    monitor="highest held-out PCK@0.05; mean pixel error breaks ties",
                )
                break
        best.restore(model)
        if validation_examples:
            metrics, predictions = evaluate(
                model,
                validation_examples,
                config,
                device,
                args.batch_size,
                sigma=args.sigma,
            )
            write_overlays(
                validation_examples, predictions, nodes, out / "pose-validation"
            )
            (out / "pose-validation/metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
            (out / "pose-validation/split.json").write_text(
                json.dumps(validation_split, indent=2)
            )
            progress.emit("pose_validation", force=True, **metrics)
        progress.emit("training_complete", force=True, selected_epoch=best.epoch)
        weights = out / "pose.pt"
        torch.save(model.state_dict(), weights)
        meta = {
            "kind": "flyscope-sleap-pose",
            "framework": "pytorch",
            "sleap_nn_version": version("sleap-nn"),
            "device": str(device),
            "architecture": "SLEAP UNet / single-instance anatomical confidence maps",
            "configuration": config,
            "weights_sha256": sha(weights),
            "source_labels_sha256": sha(args.labels),
            "source_labels": str(Path(args.labels).resolve()),
            "source_videos": [
                {"path": str(v.filename), "sha256": sha(v.filename)}
                for v in data.videos
            ],
            "examples": [e[3] for e in examples],
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "sigma": args.sigma,
            "loss": losses,
            "loss_method": "foreground-background-balanced-mse-v2",
            "validation": (
                {
                    "split": validation_split,
                    "epochs": validation_history,
                    "selected_epoch": best.epoch,
                    "monitor": "highest held-out PCK@0.05; mean pixel error breaks ties",
                }
                if validation_examples
                else None
            ),
            "augmentation": (
                "in-plane quarter rotations" if not args.no_augmentation else "none"
            ),
            "preprocessing": "centered-square-rgb-0-1-v1",
            "behavior_labels_used": False,
        }
        meta["parent_weights_sha256"] = parent["weights_sha256"] if parent else None
        (out / "pose.json").write_text(json.dumps(meta, indent=2))
        progress.emit(
            "complete", force=True, output=str(out), selected_epoch=best.epoch
        )
        return meta
    except BaseException as error:
        progress.emit(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            force=True,
            error=str(error),
        )
        raise
    finally:
        progress.close()


def infer_images(
    root, images, identities, device="cpu", minimum_peak=0.05, batch_size=8
):
    import torch
    from backend.services.suite_models import torch_setup, tensor

    root = Path(root).resolve()
    meta = read_meta(root)
    selected_device = torch_setup(device=device)
    key = (str(root), meta["weights_sha256"], str(selected_device))
    if key not in MODEL_CACHE:
        model = network(meta["configuration"]).to(selected_device)
        model.load_state_dict(
            torch.load(
                root / "pose.pt", map_location=selected_device, weights_only=True
            )
        )
        MODEL_CACHE.clear()
        MODEL_CACHE[key] = model.eval()
    model = MODEL_CACHE[key]
    size = meta["configuration"]["size"]
    output = []
    for offset in range(0, len(images), batch_size):
        batch = [
            cv2.resize(im, (size, size)) for im in images[offset : offset + batch_size]
        ]
        with torch.inference_mode():
            maps = (
                model(tensor(batch, selected_device)).permute(0, 2, 3, 1).cpu().numpy()
            )
        for identity, heatmaps in zip(identities[offset : offset + batch_size], maps):
            points = []
            for k in range(heatmaps.shape[-1]):
                y, x = np.unravel_index(np.argmax(heatmaps[..., k]), heatmaps.shape[:2])
                peak = float(heatmaps[y, x, k])
                points.append(
                    {
                        "x": float(x / size),
                        "y": float(y / size),
                        "score": peak,
                        "visible": peak >= minimum_peak,
                    }
                )
            output.append(
                {
                    "id": identity,
                    "nodes": meta["configuration"]["nodes"],
                    "points": points,
                }
            )
    return output


def predict(args):
    rows = json.loads(Path(args.request).read_text())["images"]
    images = []
    for row in rows:
        image = cv2.imread(row["path"])
        if image is None:
            raise ValueError("Unreadable pose input.")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output = infer_images(
        args.model,
        images,
        [r["id"] for r in rows],
        args.device,
        args.minimum_peak,
        args.batch_size,
    )
    Path(args.output).write_text(json.dumps(output, allow_nan=False))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    t = subs.add_parser("train")
    t.add_argument("--labels", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--size", type=int, default=128)
    t.add_argument("--filters", type=int, default=32)
    t.add_argument("--seed", type=int, default=2026)
    t.add_argument("--learning-rate", type=float, default=0.001)
    t.add_argument("--sigma", type=float, default=3)
    t.add_argument("--max-examples", type=int)
    t.add_argument("--validation-fraction", type=float, default=0.2)
    t.add_argument("--validation-gap-frames", type=int, default=10)
    t.add_argument(
        "--allow-no-validation",
        action="store_true",
        help="Explicit software-smoke opt-out; not a pose quality estimate",
    )
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--no-augmentation", action="store_true")
    t.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    t.add_argument(
        "--parent",
        help="Optional prior pose-model directory for fine-tuning with anatomical labels",
    )
    i = subs.add_parser("predict")
    i.add_argument("--model", required=True)
    i.add_argument("--request", required=True)
    i.add_argument("--output", required=True)
    i.add_argument("--batch-size", type=int, default=8)
    i.add_argument("--minimum-peak", type=float, default=0.05)
    i.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    server = subs.add_parser("serve")
    server.add_argument("--model", required=True)
    args = p.parse_args()
    if args.command == "train":
        if (
            args.epochs < 1
            or args.batch_size < 1
            or args.size < 32
            or args.size % 8
            or args.filters < 1
            or args.sigma <= 0
            or args.patience < 0
            or not 0 < args.validation_fraction < 1
            or args.validation_gap_frames < 0
        ):
            p.error(
                "Use positive training parameters and an image size divisible by 8."
            )
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                predict(
                    SimpleNamespace(
                        model=args.model,
                        request=request["request"],
                        output=request["output"],
                        batch_size=8,
                        minimum_peak=0.05,
                        device="cpu",
                    )
                )
                print(json.dumps({"ok": True}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
