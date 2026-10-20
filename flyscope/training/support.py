"""Live training records, bounded prefetch and disposable, label-free crop caches."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


class Progress:
    fields = [
        "event",
        "architecture",
        "run",
        "phase",
        "epoch",
        "epochs",
        "batch",
        "batches",
        "samples",
        "total_samples",
        "loss",
        "validation_loss",
        "validation_balanced_accuracy",
        "validation_coma_recall",
        "validation_pck_005",
        "validation_mean_error_px",
        "validation_peak_coverage",
        "elapsed_s",
        "samples_per_s",
        "input_wait_s",
        "eta_s",
        "gpu_allocated_mib",
        "gpu_reserved_mib",
    ]

    def __init__(self, output, every=5):
        self.output = Path(output)
        self.every = every
        self.last = {}
        self.epochs = []
        self.start = time.monotonic()
        self.events = (self.output / "progress.jsonl").open("w", buffering=1)
        self.table = (self.output / "training.csv").open("w", newline="")
        self.csv = csv.DictWriter(
            self.table, fieldnames=self.fields, extrasaction="ignore"
        )
        self.csv.writeheader()
        self.table.flush()

    def emit(self, event, *, force=False, **values):
        now = time.monotonic()
        key = (
            event,
            values.get("architecture"),
            values.get("run"),
            values.get("phase"),
        )
        if not force and now - self.last.get(key, -float("inf")) < self.every:
            return
        self.last[key] = now
        record = {
            "event": event,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "wall_elapsed_s": round(now - self.start, 3),
            **values,
        }
        self.events.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        self.table.flush()

    def batch(self, **values):
        # Every batch is retained for plotting; flush at progress intervals, not
        # every scalar write. Epoch rows contain the original mean batch loss.
        self.csv.writerow({"event": "batch", **values})
        self.emit("batch", **values)

    def epoch(self, **values):
        row = {"event": "epoch", **values}
        self.csv.writerow(row)
        self.epochs.append(row)
        self.emit("epoch", force=True, **values)

    def plot(self):
        self.table.flush()
        from flyscope.reporting.training_plots import plot_epochs

        # Do not rescan millions of batch rows after every epoch.
        plot_epochs(self.epochs, self.output / "loss.png")

    def close(self):
        self.table.close()
        self.events.close()


def prefetched(batches, prepare, ahead=2):
    """One CPU producer, bounded memory, unchanged batch/shuffle order.

    A thread avoids copying the annotation manifest into worker processes and
    works with the same interpreter on Linux and Windows. Exceptions propagate.
    """
    if not ahead:
        for batch in batches:
            yield batch, prepare(batch)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="flyscope-input") as pool:
        iterator = iter(batches)
        pending = deque()
        for _ in range(ahead):
            batch = next(iterator, None)
            if batch is not None:
                pending.append((batch, pool.submit(prepare, batch)))
        while pending:
            batch, future = pending.popleft()
            value = future.result()
            following = next(iterator, None)
            if following is not None:
                pending.append((following, pool.submit(prepare, following)))
            yield batch, value


def host_tensor(images, pin=False):
    import torch

    value = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy())
    return value.pin_memory() if pin else value


def device_tensor(value, device):
    # Transfer uint8 rather than a 4x larger CPU float copy. Division remains
    # float32 with exactly the same 0..1 pixel values as the inference path.
    return value.to(device, non_blocking=device.type == "cuda").float().div_(255)


class CropCache:
    """Frame-major preparation once; random-access uint8 memmap during training.

    No labels or fitted statistics are cached. Selected fit/validation wells and their causal context are requested.
    Validation pixels are cached without labels or gradient updates; test grids
    never enter this cache.
    A complete marker is published atomically after all arrays are flushed.
    """

    @classmethod
    def prepare(cls, data, runs, size, context, directory, progress, workers=2):
        from flyscope.models.runtime import PREPROCESSING, crop_rgb

        keys = set()
        for identity in {k for run in runs for k in run["annotation_ids"]}:
            dataset, annotation = data.annotations[identity]
            start = dataset.manifest.get(
                "analysis_start_index", min(data.image_maps[dataset.sha])
            )
            for frame in range(
                max(start, annotation["frame_index"] - context),
                annotation["frame_index"] + 1,
            ):
                key = (dataset.sha, frame, annotation["well_uid"], size)
                # Validate even when reusing a cache; never bypass review checks.
                data.context_geometry(*key[:3])
                keys.add(key)
        keys = sorted(keys)
        signature = {"format": 1, "preprocessing": PREPROCESSING, "keys": keys}
        token = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / token
        shape = (len(keys), size, size, 3)
        expected = len(keys) * (size * size * 3 + 4 * 8)
        progress.emit(
            "crop_cache",
            force=True,
            phase="preparation",
            samples=len(keys),
            bytes=expected,
            path=str(destination),
            reused=destination.exists(),
        )
        if not destination.exists():
            if shutil.disk_usage(root).free < expected + 16 * 1024**2:
                raise ValueError(
                    f"Crop cache needs {expected / 1024**3:.2f} GiB free at {root}; choose --cache-dir on a larger disk or --no-cache."
                )
            temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
            rgb = foreground = None
            try:
                rgb = np.lib.format.open_memmap(
                    temporary / "rgb.npy", mode="w+", dtype=np.uint8, shape=shape
                )
                foreground = np.lib.format.open_memmap(
                    temporary / "foreground.npy",
                    mode="w+",
                    dtype=np.float64,
                    shape=(len(keys), 4),
                )
                groups = {}
                for i, (sha, frame, well, _) in enumerate(keys):
                    groups.setdefault((sha, frame), []).append((i, well))
                datasets = {d.sha: d for d in data.datasets}

                def convert(group):
                    (sha, frame), wells = group
                    # This is the only full-image decode for this frame.
                    image = datasets[sha].image(data.image_maps[sha][frame]["path"])
                    for i, well in wells:
                        rgb[i], foreground[i] = crop_rgb(
                            image, data.geometry[sha][(well, frame)]["points"], size
                        )
                    return len(wells)

                completed = 0
                start_time = time.monotonic()
                # map submits eagerly in Python 3.12. Keep a bounded queue;
                # each job processes one full frame.
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pending = deque()
                    groups_iter = iter(groups.items())
                    for _ in range(workers * 2):
                        group = next(groups_iter, None)
                        if group is not None:
                            pending.append(pool.submit(convert, group))
                    while pending:
                        completed += pending.popleft().result()
                        group = next(groups_iter, None)
                        if group is not None:
                            pending.append(pool.submit(convert, group))
                        elapsed = time.monotonic() - start_time
                        progress.emit(
                            "preparing_crops",
                            phase="preparation",
                            samples=completed,
                            total_samples=len(keys),
                            samples_per_s=completed / max(elapsed, 1e-9),
                            eta_s=(len(keys) - completed) * elapsed / completed,
                        )
                rgb.flush()
                foreground.flush()
                rgb._mmap.close()
                foreground._mmap.close()
                rgb = foreground = None  # Close before rename on Windows.
                (temporary / "complete.json").write_text(
                    json.dumps({"signature": token, "samples": len(keys)})
                )
                try:
                    temporary.rename(destination)
                except OSError:
                    # Another identical training command finished first.
                    if not (destination / "complete.json").is_file():
                        raise
                    shutil.rmtree(temporary)
            finally:
                if rgb is not None:
                    rgb._mmap.close()
                if foreground is not None:
                    foreground._mmap.close()
                if temporary.exists():
                    shutil.rmtree(temporary)
        marker = json.loads((destination / "complete.json").read_text())
        if marker != {"signature": token, "samples": len(keys)}:
            raise ValueError(
                f"Invalid crop cache at {destination}; remove that cache directory and retry."
            )
        cache = cls()
        cache.index = {tuple(key): i for i, key in enumerate(keys)}
        cache.rgb = np.load(destination / "rgb.npy", mmap_mode="r")
        cache.foreground = np.load(destination / "foreground.npy", mmap_mode="r")
        if (
            cache.rgb.shape != shape
            or cache.rgb.dtype != np.uint8
            or cache.foreground.shape != (len(keys), 4)
            or cache.foreground.dtype != np.float64
        ):
            raise ValueError(
                f"Invalid crop cache arrays at {destination}; remove that cache directory and retry."
            )
        progress.emit(
            "crops_ready",
            force=True,
            phase="preparation",
            samples=len(keys),
            path=str(destination),
        )
        return cache

    def get(self, sha, frame, well, size):
        index = self.index[(sha, frame, well, size)]
        return self.rgb[index], self.foreground[index]

    def close(self):
        self.rgb._mmap.close()
        self.foreground._mmap.close()
