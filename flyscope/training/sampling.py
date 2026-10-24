"""Thin fitting targets only; keep original images and causal context intact."""

from collections import defaultdict
import cv2
import numpy as np


def changed_pixels(previous, current, delta=8):
    a = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
    difference = b - a
    # Ignore uniform illumination drift, but retain local changes such as legs
    # extending without a change in the fly's centroid.
    difference -= np.median(difference)
    return int(np.count_nonzero(np.abs(difference) >= delta))


def thin_targets(
    data, run, size, delta=8, minimum_pixels=3, keep_every=30, transition_radius=5
):
    groups = defaultdict(list)
    for identity in run["annotation_ids"]:
        dataset, annotation = data.annotations[identity]
        groups[(dataset.sha, annotation["well_uid"])].append(identity)
    retained, reasons = set(), {}
    for ids in groups.values():
        ids.sort(key=lambda k: data.annotations[k][1]["frame_index"])
        transitions = [
            data.annotations[ids[i]][1]["frame_index"]
            for i in range(1, len(ids))
            if data.annotations[ids[i]][1]["label"]
            != data.annotations[ids[i - 1]][1]["label"]
        ]
        last_image = None
        last_frame = None
        for i, identity in enumerate(ids):
            a = data.annotations[identity][1]
            # Preserve both ends of every labeled run, label transitions and
            # gaps. Human labels take precedence over a static-image heuristic.
            boundary = i in (0, len(ids) - 1) or any(
                data.annotations[ids[j]][1]["label"] != a["label"]
                or abs(data.annotations[ids[j]][1]["frame_index"] - a["frame_index"])
                != 1
                for j in (i - 1, i + 1)
                if 0 <= j < len(ids)
            )
            image = data.sample(identity, size)[0]
            moving = (
                last_image is None
                or changed_pixels(last_image, image, delta) >= minimum_pixels
            )
            dense = any(
                abs(a["frame_index"] - frame) <= transition_radius
                for frame in transitions
            )
            spaced = last_frame is None or a["frame_index"] - last_frame >= keep_every
            if boundary or moving or dense or spaced:
                retained.add(identity)
                last_image = image
                last_frame = a["frame_index"]
                reasons[identity] = (
                    "label_or_sequence_boundary"
                    if boundary
                    else (
                        "transition_window"
                        if dense
                        else "spaced_static_anchor" if spaced else "pixel_change"
                    )
                )
    selected = [k for k in run["annotation_ids"] if k in retained]
    sampling = {
        "method": "per-well-last-retained-pixel-change-v1",
        "delta": delta,
        "minimum_pixels": minimum_pixels,
        "image_size": size,
        "keep_every_frames": keep_every,
        "transition_radius_frames": transition_radius,
        "original_targets": len(run["annotation_ids"]),
        "retained_targets": len(selected),
        "removed_annotation_ids": [
            k for k in run["annotation_ids"] if k not in retained
        ],
        "retention_reasons": reasons,
        "validation_thinned": False,
        "context_images_removed": False,
    }
    return {**run, "annotation_ids": selected}, sampling
