import json
import math
from pathlib import Path
import tempfile
import unittest

from flyscope.data.format import hydrate_manifest
from flyscope.imaging.geometry import clip_polygon, is_simple_polygon
from flyscope.data.datasets import Dataset


class CompactDatasets(unittest.TestCase):
    def test_shared_geometry_keeps_overrides_partial_flags_and_unlabeled_context(self):
        base = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
        changed = [[0.2, 0.1], [0.5, 0.1], [0.5, 0.4], [0.2, 0.4]]
        m = dict(
            geometry_format="shared-well-boundaries-v1",
            task="ccrt",
            split_group="one",
            images=[
                dict(
                    frame_index=i,
                    frame_number=i + 1,
                    path=f"images/{i}.png",
                    width=100,
                    height=100,
                    original_filename=f"{i}.png",
                )
                for i in range(3)
            ],
            wells=[
                dict(
                    well_uid="a",
                    well_id="W1",
                    points=base,
                    overrides={"1": changed},
                    review_frames=[1],
                    partial_frames={"2": True},
                )
            ],
            annotations=[dict(id="a1", frame_index=1, well_uid="a", label="coma")],
        )
        hydrate_manifest(m)
        self.assertIs(m["annotations"][0]["boundary_normalized"], changed)
        self.assertEqual(m["annotations"][0]["image_path"], "images/1.png")
        frames = m["wells"][0]["frames"]
        self.assertEqual(len(frames), 3)
        self.assertIs(frames[0]["points"], frames[2]["points"])
        self.assertFalse(frames[1]["boundary_reviewed"])
        self.assertTrue(frames[2]["partial"])
        self.assertEqual(len(m["annotations"]), 1)

    def test_incomplete_folder_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "annotations.json").write_text(
                json.dumps(dict(completion_marker="dataset-complete.json"))
            )
            with self.assertRaisesRegex(ValueError, "Incomplete dataset"):
                Dataset(root, "train")

    def test_tangent_clipping_never_introduces_image_corner_vertices(self):
        # Border wells with near-tangent edges previously hit OpenCV float32
        # intersection degeneracies. The clipped shape must remain in its box.
        for angle in [0, 0.0000001, 0.1, 0.37]:
            points = [
                [
                    0.3425 + 0.028 * math.cos(a + angle),
                    0.976 + 0.039 * math.sin(a + angle),
                ]
                for a in [i * 2 * math.pi / 40 for i in range(40)]
            ]
            clipped = clip_polygon(points)
            self.assertTrue(is_simple_polygon(clipped))
            self.assertTrue(all(0.31 < x < 0.38 and 0.93 < y <= 1 for x, y in clipped))


if __name__ == "__main__":
    unittest.main()
