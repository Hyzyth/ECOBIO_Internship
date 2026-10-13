#!/usr/bin/env python3
"""Isolated Python 3.10 SLEAP pose pretraining and batched keypoint inference.

Native SLP annotations provide anatomical targets, never coma/awake labels.
The exported confidence-map model uses SLEAP's actual UNet backbone. Keeping
this worker separate avoids conflicting TensorFlow/Keras versions in the app.
"""
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flyscope-sleap-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/flyscope-sleap-cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
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
    import tensorflow as tf
    from sleap.nn.architectures.unet import UNet

    x = tf.keras.Input((config["size"], config["size"], 3))
    features, _ = UNet(
        filters=config["filters"], down_blocks=3, up_blocks=3
    ).make_backbone(x)
    heatmaps = tf.keras.layers.Conv2D(
        len(config["nodes"]), 1, name="anatomical_confidence_maps"
    )(features)
    return tf.keras.Model(x, heatmaps)


def centered_examples(labels, size):
    import sleap

    source = Path(labels).resolve()

    # Resolve moved videos explicitly through SLEAP's supported callback, rather
    # than consulting GUI preferences in the user's home directory.
    def locate(videos):
        for video in videos:
            backend = video["backend"]
            filename = backend["filename"]
            if not Path(filename).is_file():
                candidate = (
                    source.parent / str(filename).replace("\\", "/").split("/")[-1]
                )
                if not candidate.is_file():
                    raise ValueError(
                        f"SLP video is missing: {candidate.name}. Keep its video next to the labels."
                    )
                backend["filename"] = str(candidate)

    data = sleap.Labels.load_file(str(source), video_search=locate)
    if len(data.skeletons) != 1:
        raise ValueError("Use one anatomical skeleton per pose model.")
    nodes = data.skeletons[0].node_names
    examples = []
    image_cache = {}
    for frame in data.labeled_frames:
        if not frame.user_instances:
            continue
        key = (frame.video.filename, frame.frame_idx)
        if key not in image_cache:
            image_cache[key] = frame.video[frame.frame_idx][0]
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
                        "frame": frame.frame_idx,
                        "instance": n,
                    },
                )
            )
    if not examples:
        raise ValueError(
            "Native labels contain no usable user keypoints; predicted instances are not targets."
        )
    return nodes, examples, data


def train(args):
    import sleap
    import tensorflow as tf

    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError("Pose output must be a new directory.")
    tf.keras.utils.set_random_seed(args.seed)
    parent = None
    if args.parent:
        parent = json.loads((Path(args.parent) / "pose.json").read_text())
        args.size = parent["configuration"]["size"]
        args.filters = parent["configuration"]["filters"]
    nodes, examples, data = centered_examples(args.labels, args.size)
    if args.max_examples:
        order = np.random.default_rng(args.seed).permutation(len(examples))[
            : args.max_examples
        ]
        examples = [examples[i] for i in order]
    config = {"size": args.size, "filters": args.filters, "nodes": nodes}
    model = network(config)
    if parent:
        if (
            parent["configuration"]["nodes"] != nodes
            or sha(Path(args.parent) / "pose.weights.h5") != parent["weights_sha256"]
        ):
            raise ValueError(
                "Parent pose skeleton or weights hash differs from the provided labels."
            )
        model.load_weights(str(Path(args.parent) / "pose.weights.h5"))
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    yy, xx = np.mgrid[: args.size, : args.size]
    losses = []
    for epoch in range(args.epochs):
        epoch_loss = []
        order = np.random.default_rng(args.seed + epoch).permutation(len(examples))
        for offset in range(0, len(order), args.batch_size):
            batch = [examples[i] for i in order[offset : offset + args.batch_size]]
            x = np.stack([e[0] for e in batch]).astype(np.float32) / 255
            targets, weights = [], []
            for _, points, visible, _ in batch:
                heatmaps = np.exp(
                    -(
                        (xx[..., None] - np.nan_to_num(points[:, 0])) ** 2
                        + (yy[..., None] - np.nan_to_num(points[:, 1])) ** 2
                    )
                    / (2 * args.sigma**2)
                )
                targets.append(heatmaps)
                weights.append(visible.astype(np.float32))
            target = tf.constant(np.stack(targets), tf.float32)
            mask = tf.constant(np.stack(weights)[:, None, None, :], tf.float32)
            with tf.GradientTape() as tape:
                predicted = model(x, training=True)
                loss = tf.reduce_sum(tf.square(predicted - target) * mask) / (
                    tf.reduce_sum(mask) * args.size**2
                )
            if not np.isfinite(float(loss)):
                raise RuntimeError("Nonfinite pose loss; no artifact saved.")
            optimizer.apply_gradients(
                zip(
                    tape.gradient(loss, model.trainable_weights),
                    model.trainable_weights,
                )
            )
            epoch_loss.append(float(loss))
        losses.append(float(np.mean(epoch_loss)))
        print(json.dumps({"pose_epoch": epoch + 1, "loss": losses[-1]}), flush=True)
    out.mkdir(parents=True)
    weights = out / "pose.weights.h5"
    model.save_weights(str(weights))
    meta = {
        "kind": "flyscope-sleap-pose",
        "sleap_version": sleap.__version__,
        "architecture": "SLEAP UNet / single-instance anatomical confidence maps",
        "configuration": config,
        "weights_sha256": sha(weights),
        "source_labels_sha256": sha(args.labels),
        "source_labels": str(Path(args.labels).resolve()),
        "source_videos": [
            {"path": str(v.filename), "sha256": sha(v.filename)} for v in data.videos
        ],
        "examples": [e[3] for e in examples],
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "sigma": args.sigma,
        "loss": losses,
        "preprocessing": "centered-square-rgb-0-1-v1",
        "behavior_labels_used": False,
    }
    meta["parent_weights_sha256"] = parent["weights_sha256"] if parent else None
    (out / "pose.json").write_text(json.dumps(meta, indent=2))
    return meta


def predict(args):
    root = Path(args.model).resolve()
    meta = json.loads((root / "pose.json").read_text())
    if sha(root / "pose.weights.h5") != meta["weights_sha256"]:
        raise ValueError("Pose weights do not match their metadata hash.")
    key = (str(root), meta["weights_sha256"])
    if key not in MODEL_CACHE:
        model = network(meta["configuration"])
        model.load_weights(str(root / "pose.weights.h5"))
        MODEL_CACHE.clear()
        MODEL_CACHE[key] = model
    model = MODEL_CACHE[key]
    size = meta["configuration"]["size"]
    rows = json.loads(Path(args.request).read_text())["images"]
    output = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        x = []
        for row in batch:
            image = cv2.imread(row["path"])
            if image is None:
                raise ValueError("Unreadable pose input.")
            x.append(cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB))
        maps = model(np.stack(x).astype(np.float32) / 255, training=False).numpy()
        for row, heatmaps in zip(batch, maps):
            points = []
            for k in range(heatmaps.shape[-1]):
                y, x0 = np.unravel_index(
                    np.argmax(heatmaps[..., k]), heatmaps.shape[:2]
                )
                peak = float(heatmaps[y, x0, k])
                points.append(
                    {
                        "x": float(x0 / size),
                        "y": float(y / size),
                        "score": peak,
                        "visible": peak >= args.minimum_peak,
                    }
                )
            output.append(
                {
                    "id": row["id"],
                    "nodes": meta["configuration"]["nodes"],
                    "points": points,
                }
            )
    Path(args.output).write_text(json.dumps(output))
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
                    )
                )
                print(json.dumps({"ok": True}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
