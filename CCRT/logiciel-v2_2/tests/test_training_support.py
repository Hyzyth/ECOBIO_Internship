"""Input equivalence, causal caching and live records for long training jobs."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.suite_fixtures import grids
from tools.model_suite import TrainingData, main, plan_for
from tools.training_support import (
    CropCache,
    Progress,
    device_tensor,
    host_tensor,
    prefetched,
)
from backend.services.suite_models import tensor


class TrainingSupportTests(unittest.TestCase):
    def test_cached_crops_match_original_and_decode_once_per_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, _ = grids(root / "fixtures")
            data = TrainingData([train])
            run = plan_for(data)["runs"][0]
            expected = {k: data.sample(k, 64) for k in run["annotation_ids"]}
            with patch.object(
                data.datasets[0], "image", wraps=data.datasets[0].image
            ) as decode:
                progress = Progress(root)
                try:
                    data.cache = CropCache.prepare(
                        data, [run], 64, 4, root / "cache", progress
                    )
                    self.assertEqual(decode.call_count, 6)
                    data.rgb.cache_clear()
                    for key, (rgb, foreground) in expected.items():
                        actual = data.sample(key, 64)
                        np.testing.assert_array_equal(actual[0], rgb)
                        np.testing.assert_array_equal(actual[1], foreground)
                    # Repeated epochs and a new command reuse the same disk pixels.
                    CropCache.prepare(data, [run], 64, 4, root / "cache", progress)
                    self.assertEqual(decode.call_count, 6)
                finally:
                    cache = data.cache
                    data.cache = None
                    data.rgb.cache_clear()
                    if cache is not None:
                        cache.close()
                    progress.close()

    def test_cache_includes_only_selected_well_and_causal_analysis_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, _ = grids(root / "fixtures")
            data = TrainingData([train])
            data.datasets[0].manifest["analysis_start_index"] = 2
            key = next(
                k for k, (_, a) in data.annotations.items() if a["frame_index"] == 5
            )
            dataset, annotation = data.annotations[key]
            progress = Progress(root)
            try:
                cache = CropCache.prepare(
                    data,
                    [{"annotation_ids": [key]}],
                    64,
                    4,
                    root / "cache",
                    progress,
                    workers=1,
                )
                self.assertEqual({k[1] for k in cache.index}, {2, 3, 4, 5})
                self.assertEqual({k[2] for k in cache.index}, {annotation["well_uid"]})
                # Even an already built cache must not bypass geometry review.
                data.geometry[dataset.sha][(annotation["well_uid"], 3)][
                    "boundary_reviewed"
                ] = False
                with self.assertRaisesRegex(ValueError, "boundary needs review"):
                    CropCache.prepare(
                        data,
                        [{"annotation_ids": [key]}],
                        64,
                        4,
                        root / "cache",
                        progress,
                    )
                del cache
            finally:
                progress.close()

    def test_prefetch_preserves_order_and_propagates_input_failure(self):
        batches = [[3, 1], [9], [2, 0]]
        for ahead in [0, 1, 2]:
            self.assertEqual(
                list(prefetched(iter(batches), sum, ahead)),
                [(b, sum(b)) for b in batches],
            )

        def fail(batch):
            raise ValueError("unreadable context")

        with self.assertRaisesRegex(ValueError, "unreadable context"):
            list(prefetched(batches, fail))

    def test_failed_preparation_does_not_publish_or_retain_partial_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, _ = grids(root / "fixtures")
            data = TrainingData([train])
            run = plan_for(data)["runs"][0]
            progress = Progress(root)
            try:
                with patch.object(
                    data.datasets[0], "image", side_effect=ValueError("decode failed")
                ):
                    with self.assertRaisesRegex(ValueError, "decode failed"):
                        CropCache.prepare(data, [run], 64, 0, root / "cache", progress)
                self.assertEqual(list((root / "cache").iterdir()), [])
            finally:
                progress.close()

    def test_uint8_transfer_keeps_the_original_pixel_values(self):
        import torch

        images = np.random.default_rng(22).integers(
            0, 256, (3, 32, 32, 3), dtype=np.uint8
        )
        actual = device_tensor(host_tensor(images), torch.device("cpu"))
        np.testing.assert_array_equal(
            actual.numpy(), tensor(images, torch.device("cpu")).numpy()
        )

    def test_live_losses_are_readable_before_completion_and_plot_epochs(self):
        from tools.plot_training import plot_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = Progress(root)
            try:
                progress.batch(
                    architecture="resnet18",
                    run="poor-poses",
                    phase="appearance",
                    epoch=1,
                    loss=0.9,
                )
                with (root / "training.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 1)
                self.assertFalse(
                    plot_training(root / "training.csv", root / "loss.png")
                )
                progress.epoch(
                    architecture="resnet18",
                    run="poor-poses",
                    phase="appearance",
                    epoch=1,
                    loss=0.7,
                )
                progress.plot()
                self.assertGreater((root / "loss.png").stat().st_size, 1000)
                records = [
                    json.loads(row)
                    for row in (root / "progress.jsonl").read_text().splitlines()
                ]
                self.assertEqual([r["event"] for r in records], ["batch", "epoch"])
                self.assertEqual(records[-1]["loss"], 0.7)
            finally:
                progress.close()

    def test_training_failure_keeps_live_logs_and_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, test = grids(root / "fixtures")
            out = root / "results"
            with patch(
                "tools.model_suite.train_model",
                side_effect=RuntimeError("training interrupted here"),
            ):
                with self.assertRaisesRegex(RuntimeError, "training interrupted here"):
                    main(
                        [
                            "compare",
                            "--train",
                            str(train),
                            "--test",
                            str(test),
                            "--architectures",
                            "resnet18",
                            "--allow-no-validation",
                            "--no-cache",
                            "--output",
                            str(out),
                        ]
                    )
            events = [
                json.loads(row)
                for row in (out / "progress.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[-1]["event"], "failed")
            self.assertEqual(events[-1]["error_type"], "RuntimeError")
            self.assertEqual(events[-1]["error"], "training interrupted here")


if __name__ == "__main__":
    unittest.main()
