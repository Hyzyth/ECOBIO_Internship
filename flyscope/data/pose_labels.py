"""Human anatomical targets on exact CCRT pose inputs; behavior labels are context."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from .datasets import Dataset, safe_path

NODES = [
    "head",
    "thorax",
    "abdomen",
    "wingL",
    "wingR",
    "forelegL4",
    "forelegR4",
    "midlegL4",
    "midlegR4",
    "hindlegL4",
    "hindlegR4",
    "eyeL",
    "eyeR",
]


def prepare(dataset, output, count=48, size=128, skeleton=None, wells=None):
    from flyscope.models.runtime import crop_rgb, pose_input, POSE_PREPROCESSING

    data = Dataset(dataset, "train")
    out = Path(output)
    if out.exists():
        raise ValueError("Use a new pose-label project directory.")
    if count < 2 or size < 32:
        raise ValueError("Use at least two examples and size >=32.")
    nodes = (
        json.loads(Path(skeleton).read_text())["configuration"]["nodes"]
        if skeleton
        else NODES
    )
    labels = {
        (a["well_uid"], a["frame_index"]): a["label"]
        for a in data.annotations.values()
        if a.get("eligible_for_training")
    }
    groups = defaultdict(list)
    for w in data.manifest["wells"]:
        if (
            w.get("occupancy") != "single"
            or wells
            and w["well_id"] not in wells
            and w["well_uid"] not in wells
        ):
            continue
        for frame in w.get("frames", []):
            if frame.get("partial") or frame.get("boundary_reviewed") is False:
                continue
            state = labels.get((w["well_uid"], frame["frame_index"]), "unannotated")
            groups[(state, w["well_uid"])].append((w, frame))
    # Even temporal coverage within each well/state, then round-robin across
    # states and individuals. No predictions or test-grid selection is used.
    queues = defaultdict(list)
    for (state, uid), values in sorted(groups.items()):
        ordered = sorted(values, key=lambda v: v[1]["frame_index"])
        indices = np.linspace(0, len(ordered) - 1, min(len(ordered), count), dtype=int)
        queues[state].append((uid, [ordered[i] for i in indices]))
    chosen = []
    while len(chosen) < count and any(q for state in queues.values() for _, q in state):
        for state, individuals in sorted(queues.items()):
            if len(chosen) >= count:
                break
            individual = next(((uid, q) for uid, q in individuals if q), None)
            if individual:
                uid, queue = individual
                chosen.append(((state, uid), *queue.pop(0)))
                individuals.remove(individual)
                individuals.append(individual)
    if not chosen:
        raise ValueError(
            "No reviewed single-individual wells match. Check occupancy and boundary review in the CCRT app."
        )
    out.mkdir(parents=True)
    (out / "images").mkdir()
    images = {im["frame_index"]: im for im in data.manifest["images"]}
    examples = []
    for index, ((state, uid), w, frame) in enumerate(chosen):
        source = images[frame["frame_index"]]
        rgb, _ = crop_rgb(data.image(source["path"]), frame["points"], size)
        rgb = pose_input(rgb)
        relative = f"images/{index+1:05d}.png"
        path = out / relative
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        group = f"{data.manifest['experiment_id']}::{uid}"
        examples.append(
            {
                "id": f'{uid}:{frame["frame_index"]}',
                "image_path": relative,
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "group_id": group,
                "source_dataset_sha256": data.sha,
                "source_experiment": data.manifest["experiment_id"],
                "source_image_sha256": source.get("source_sha256", source["sha256"]),
                "frame": frame["frame_index"],
                "well_id": w["well_id"],
                "state_context": state,
                "reviewed": False,
                "points": [
                    {"status": "unlabeled", "x": None, "y": None} for _ in nodes
                ],
            }
        )
    manifest = {
        "kind": "flyscope-pose-labels",
        "schema_version": 1,
        "nodes": nodes,
        "size": size,
        "preprocessing": POSE_PREPROCESSING,
        "examples": examples,
        "instructions": "Coordinates are normalized on the exported exact pose input. Only manually reviewed visible keypoints train the estimator; occluded points are excluded. Coma/Awake context is never an anatomical target. Hold out complete individuals.",
    }
    (out / "pose-labels.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.txt").write_text(
        "Open the app Pose annotation page, select this entire folder, mark visible points or explicitly occluded nodes, and confirm reviewed examples. Export the labeled dataset ZIP, extract it, then run tools/sleap_pose.py train --labels PATH/pose-labels.json --parent PATH/pose-base --output NEW_MODEL. Keep at least two independent wells for grouped validation. Do not use a final test grid for fine-tuning.\n"
    )
    return manifest


def load_examples(labels, size):
    source = Path(labels).resolve()
    meta = json.loads(source.read_text())
    if meta.get("kind") != "flyscope-pose-labels" or meta.get("schema_version") != 1:
        raise ValueError("Not a FlyScope anatomical annotation project.")
    from flyscope.models.runtime import POSE_PREPROCESSING

    if meta.get("preprocessing") != POSE_PREPROCESSING:
        raise ValueError("Pose annotation preprocessing differs from runtime inputs.")
    nodes = meta.get("nodes", [])
    if (
        not nodes
        or len(set(nodes)) != len(nodes)
        or not all(isinstance(n, str) and n for n in nodes)
    ):
        raise ValueError("Node names must be nonempty and unique.")
    examples = []
    seen = set()
    for e in meta["examples"]:
        if e["id"] in seen:
            raise ValueError("Duplicate pose example ID.")
        seen.add(e["id"])
        if e.get("reviewed") is not True:
            continue
        if not e.get("group_id"):
            raise ValueError(
                "Pose labels need stable individual identities for validation."
            )
        path = safe_path(source.parent, e["image_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != e["image_sha256"]:
            raise ValueError("Pose source image changed since annotation.")
        if len(e["points"]) != len(nodes):
            raise ValueError("Pose point count differs from the skeleton.")
        xy = np.full((len(nodes), 2), np.nan)
        visible = np.zeros(len(nodes), bool)
        for i, p in enumerate(e["points"]):
            if p["status"] == "visible":
                v = [p.get("x"), p.get("y")]
                if any(
                    isinstance(n, bool)
                    or not isinstance(n, (float, int))
                    or not np.isfinite(n)
                    or not 0 <= n <= 1
                    for n in v
                ):
                    raise ValueError(
                        "Visible keypoints need finite normalized coordinates."
                    )
                xy[i] = np.asarray(v) * (size - 1)
                visible[i] = True
            elif p["status"] != "occluded":
                raise ValueError(
                    "Reviewed examples must explicitly label or occlude every node."
                )
        if visible.sum() < 2:
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Unreadable pose input.")
        identity = {
            k: e[k]
            for k in [
                "id",
                "group_id",
                "frame",
                "well_id",
                "source_experiment",
                "source_image_sha256",
            ]
        }
        identity.update(video=str(path), video_sha256=e["image_sha256"], instance=0)
        examples.append(
            (
                cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB),
                xy,
                visible,
                identity,
            )
        )
    if not examples:
        raise ValueError(
            "No reviewed examples with at least two visible keypoints. Annotate them in the Pose annotation page first."
        )
    return nodes, examples, SimpleNamespace(videos=[])
