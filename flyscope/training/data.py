"""Selected manual targets and causal, same-individual pixel access."""

from functools import lru_cache
from flyscope.data.datasets import Dataset
from flyscope.models.runtime import crop_rgb


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
