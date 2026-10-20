"""Verified CCRT exports, whole-well plans and independent-test checks.

No framework is imported for manifest or source-image inspection. The legacy
CNN sample method loads its preprocessing only when explicitly requested.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import cv2
from .format import hydrate_manifest


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Dataset image path leaves dataset directory or is missing.")
    return path


class Dataset:
    def __init__(self, root, role, progress=None):
        self.root = Path(root).resolve()
        if progress:
            progress("loading_dataset", force=True, dataset=str(self.root), role=role)
        raw = (self.root / "annotations.json").read_bytes()
        self.sha = digest(raw)
        self.manifest = json.loads(raw)
        marker = self.manifest.get("completion_marker")
        if marker is not None:
            if marker != "dataset-complete.json" or not (self.root / marker).is_file():
                raise ValueError(
                    "Incomplete dataset export. Save/extract every part, or export again."
                )
        hydrate_manifest(self.manifest)
        if self.manifest.get("role", "train") != role:
            raise ValueError(f"Expected a {role} dataset, not another role.")
        if self.manifest.get("task") != "ccrt":
            raise ValueError(
                "These trainers support CCRT; egg format remains provisional."
            )
        self.role = role
        self.images = {image["path"]: image for image in self.manifest["images"]}
        self.image_hashes = set()
        if progress:
            progress(
                "verifying_images",
                force=True,
                dataset=str(self.root),
                images=0,
                total_images=len(self.images),
            )
        for index, (relative, image) in enumerate(self.images.items(), 1):
            actual = digest(safe_path(self.root, relative).read_bytes())
            if actual != image["sha256"]:
                raise ValueError("Image content does not match the export hash.")
            self.image_hashes.add(actual)
            if progress:
                progress(
                    "verifying_images",
                    force=index == len(self.images),
                    dataset=str(self.root),
                    images=index,
                    total_images=len(self.images),
                )
        self.annotations = {}
        for a in self.manifest["annotations"]:
            if a["id"] in self.annotations:
                raise ValueError("Duplicate annotation IDs.")
            if a["image_path"] not in self.images:
                raise ValueError("Missing annotation image.")
            if a["eligible_for_training"] and (
                a["label"] not in ("coma", "awake", "empty")
                or a.get("label_source") not in ("manual", "manual_occupancy")
                or a.get("occupancy") not in ("empty", "single")
            ):
                raise ValueError(
                    "Eligible targets must come from suitable manual annotations."
                )
            self.annotations[a["id"]] = a

    @property
    def geometry(self):
        return {
            (w["well_uid"], f["frame_index"]): f
            for w in self.manifest.get("wells", [])
            for f in w.get("frames", [])
            if f.get("active", True)
        }

    @lru_cache(maxsize=2)
    def image(self, path):
        value = cv2.imread(str(safe_path(self.root, path)), cv2.IMREAD_COLOR)
        if value is None:
            raise ValueError("Unreadable image.")
        return value

    @lru_cache(maxsize=64)
    def sample(self, identity):
        from flyscope.models.keras import crop_well, preprocess, occupancy_features

        a = self.annotations[identity]
        crop, mask = crop_well(self.image(a["image_path"]), a["boundary_normalized"])
        features, _ = occupancy_features(crop, mask)
        return preprocess(crop, mask), features


def load_plan(dataset, path):
    plan = read_json(path)
    if (
        plan.get("kind") != "flyscope-comparison-plan"
        or plan.get("dataset_sha256") != dataset.sha
    ):
        raise ValueError(
            "Comparison plan is bound to a different dataset. Use its original export."
        )
    names = set()
    for run in plan["runs"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run["id"]) or run["id"] in names:
            raise ValueError("Invalid or duplicate run IDs.")
        names.add(run["id"])
        if not isinstance(run["seed"], int) or not 0 <= run["seed"] <= 0xFFFFFFFF:
            raise ValueError("Invalid run seed.")
        selected = run["annotation_ids"]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Empty or duplicate sample selection.")
        expected = {
            a["id"]
            for a in dataset.annotations.values()
            if a["eligible_for_training"] and a["well_uid"] in run["well_uids"]
        }
        if set(selected) != expected:
            raise ValueError(
                "Run must use exactly the eligible labels of its saved wells."
            )
    return plan


def check_holdouts(train, tests):
    for data in tests:
        if data.manifest.get("experiment_id") == train.manifest.get("experiment_id"):
            raise ValueError(
                "Training and test datasets have the same experiment/split group. Use another test grid."
            )
        if train.image_hashes & data.image_hashes:
            raise ValueError("Training and test images overlap by content hash.")
