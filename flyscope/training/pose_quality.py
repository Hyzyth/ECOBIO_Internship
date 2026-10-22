"""Pose loss, grouped validation and visible localization diagnostics.

Heatmap MSE alone cannot establish keypoint accuracy: almost every pixel is
background. PCK and overlays here are evaluated only against anatomical labels.
"""

from pathlib import Path
import json
import random

import numpy as np


def split_examples(examples, fraction=0.2, seed=2026, gap=10):
    if not 0 < fraction < 1 or gap < 0:
        raise ValueError(
            "Use a pose validation fraction in (0,1) and a nonnegative temporal gap."
        )

    def video_id(example):
        return example[3].get("group_id") or example[3].get(
            "video_sha256", example[3]["video"]
        )

    videos = sorted({video_id(e) for e in examples})
    rng = random.Random(seed)
    if len(videos) > 1:
        rng.shuffle(videos)
        chosen = set(
            videos[: max(1, min(len(videos) - 1, round(len(videos) * fraction)))]
        )
        validation = [e for e in examples if video_id(e) in chosen]
        fit = [e for e in examples if video_id(e) not in chosen]
        unit, purged = (
            "whole individual" if examples[0][3].get("group_id") else "whole video"
        ), []
    else:
        if examples[0][3].get("group_id"):
            raise ValueError(
                "Anatomical CCRT validation needs at least two independently annotated individuals; more frames of one fly are not independent validation."
            )
        frames = sorted({e[3]["frame"] for e in examples})
        if len(frames) < 2:
            raise ValueError(
                "Pose validation needs at least two distinct annotated frames, preferably independent videos."
            )
        count = max(1, min(len(frames) - 1, round(len(frames) * fraction)))
        start = rng.randrange(len(frames) - count + 1)
        selected = set(frames[start : start + count])
        low, high = min(selected) - gap, max(selected) + gap
        validation = [e for e in examples if e[3]["frame"] in selected]
        fit = [e for e in examples if not low <= e[3]["frame"] <= high]
        purged = [
            e[3]
            for e in examples
            if e[3]["frame"] not in selected and low <= e[3]["frame"] <= high
        ]
        unit = "contiguous annotated time block in one video; same animals may recur"
    if not fit or not validation:
        raise ValueError(
            "Pose validation leaves no fit examples. Add independent labels, reduce --validation-gap-frames, or explicitly use --allow-no-validation for a software smoke test."
        )
    return (
        fit,
        validation,
        {
            "unit": unit,
            "gap_frames": gap,
            "training_examples": [e[3] for e in fit],
            "validation_examples": [e[3] for e in validation],
            "purged_examples": purged,
        },
    )


def heatmap_loss(predicted, target, visible):
    """Equal foreground/background contributions, averaged over visible nodes."""
    error = (predicted - target).square()
    foreground = (target >= 0.05).to(predicted.dtype) * visible
    background = (target < 0.05).to(predicted.dtype) * visible
    return 0.5 * (
        (error * foreground).sum() / foreground.sum().clamp_min(1)
        + (error * background).sum() / background.sum().clamp_min(1)
    )


def heatmap_targets(examples, size, sigma, device):
    import torch

    yy, xx = np.mgrid[:size, :size]
    maps, weights = [], []
    for _, points, visible, _ in examples:
        maps.append(
            np.exp(
                -(
                    (xx[..., None] - np.nan_to_num(points[:, 0])) ** 2
                    + (yy[..., None] - np.nan_to_num(points[:, 1])) ** 2
                )
                / (2 * sigma**2)
            )
        )
        weights.append(visible.astype(np.float32))
    return (
        torch.as_tensor(
            np.stack(maps).transpose(0, 3, 1, 2).copy(),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            np.stack(weights)[:, :, None, None], dtype=torch.float32, device=device
        ),
    )


def augment_example(example, rng):
    image, points, visible, identity = example
    turns = int(rng.integers(0, 4))
    points = points.copy()
    for _ in range(turns):
        points = np.c_[points[:, 1], image.shape[1] - 1 - points[:, 0]]
    # Rotation preserves anatomical node names; no guessed left/right swaps.
    return np.rot90(image, turns).copy(), points, visible, identity


def evaluate(
    network, examples, config, device, batch_size=8, minimum_peak=0.05, sigma=3
):
    import torch
    from flyscope.models.runtime import tensor

    nodes, size = config["nodes"], config["size"]
    rows, distances, correct, covered, visible_counts, losses = [], [], [], [], [], []
    network.eval()
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset : offset + batch_size]
            predicted = network(tensor([e[0] for e in batch], device))
            targets, mask = heatmap_targets(batch, size, sigma, device)
            losses.append(float(heatmap_loss(predicted, targets, mask)) * len(batch))
            maps = predicted.cpu().numpy()
            for example, heatmaps in zip(batch, maps):
                flat = heatmaps.reshape(len(nodes), -1)
                maxima = flat.argmax(1)
                xy = np.c_[maxima % size, maxima // size]
                peaks = flat[np.arange(len(nodes)), maxima]
                gt, visible = example[1], example[2]
                errors = np.linalg.norm(xy - gt, axis=1)
                found = peaks >= minimum_peak
                distances.append(np.where(visible, errors, np.nan))
                correct.append(visible & found & (errors <= 0.05 * size))
                covered.append(visible & found)
                visible_counts.append(visible)
                rows.append(
                    {
                        "identity": example[3],
                        "predicted_xy": xy.tolist(),
                        "scores": peaks.tolist(),
                        "ground_truth_xy": np.where(np.isfinite(gt), gt, -1).tolist(),
                        "visible": visible.tolist(),
                    }
                )
    if not rows:
        raise ValueError("No pose examples to evaluate.")
    errors = np.stack(distances)
    counts = np.stack(visible_counts).sum(0)
    hits = np.stack(correct).sum(0)
    valid = np.stack(covered).sum(0)
    total = int(counts.sum())
    metrics = {
        "examples": len(rows),
        "visible_targets": total,
        "balanced_heatmap_loss": sum(losses) / len(examples),
        "mean_error_px": float(np.nanmean(errors)),
        "median_error_px": float(np.nanmedian(errors)),
        "pck_005": float(hits.sum() / total),
        "visible_prediction_coverage": float(valid.sum() / total),
        "definition": "PCK@0.05: predicted peak >=0.05 AND within 5% of the square input side, divided by visible anatomical targets. Pixel errors use the model input size.",
        "nodes": {
            node: {
                "targets": int(n),
                "pck_005": float(h / n) if n else None,
                "coverage": float(v / n) if n else None,
            }
            for node, n, h, v in zip(nodes, counts, hits, valid)
        },
    }
    return metrics, rows


def write_overlays(examples, rows, nodes, output, maximum=24):
    from PIL import Image, ImageDraw

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(rows) - 1, min(maximum, len(rows)), dtype=int)
    tiles = []
    for index in indices:
        image = Image.fromarray(examples[index][0]).resize((320, 320))
        panel = Image.new("RGB", (320, 365), "#182832")
        panel.paste(image, (0, 45))
        draw = ImageDraw.Draw(panel)
        row = rows[index]
        caption = (
            row["identity"].get("caption")
            or f"Frame {row['identity'].get('frame')} / instance {row['identity'].get('instance')}"
        )
        draw.text((6, 5), caption[:48], fill="white")
        draw.text((6, 23), "Predicted: magenta | Human: green crosses", fill="white")
        scale = 320 / examples[index][0].shape[0]
        for node, (xy, score, gt, visible) in enumerate(
            zip(
                row["predicted_xy"],
                row["scores"],
                row["ground_truth_xy"],
                row["visible"],
            ),
            1,
        ):
            x, y = xy[0] * scale, 45 + xy[1] * scale
            color = "#ff3ec8" if score >= 0.05 else "#bbbbbb"
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            draw.text((x + 4, y), str(node), fill=color)
            if visible:
                x, y = gt[0] * scale, 45 + gt[1] * scale
                draw.line((x - 5, y, x + 5, y), fill="#00ff66", width=2)
                draw.line((x, y - 5, x, y + 5), fill="#00ff66", width=2)
        tiles.append(panel)
    columns = min(3, len(tiles))
    sheet = Image.new(
        "RGB", (columns * 320, ((len(tiles) + columns - 1) // columns) * 365), "#182832"
    )
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * 320, (i // columns) * 365))
    sheet.save(output / "overlays.jpg", quality=90)
    (output / "nodes.json").write_text(
        json.dumps({str(i + 1): node for i, node in enumerate(nodes)}, indent=2)
    )
