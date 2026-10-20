"""Synthetic, distinct train/test grids for real-framework workflow checks."""

import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def grids(root):
    root = Path(root)
    for role in ["train", "test"]:
        output = root / role
        (output / "images").mkdir(parents=True)
        images = []
        annotations = []
        wells = []
        for well in range(6):
            x = 100 + (well % 3) * 200
            y = 100 + (well // 3) * 200
            points = [
                [(x - 65) / 600, (y - 65) / 400],
                [(x + 65) / 600, (y - 65) / 400],
                [(x + 65) / 600, (y + 65) / 400],
                [(x - 65) / 600, (y + 65) / 400],
            ]
            wells.append(
                {
                    "well_uid": f"well-{well}",
                    "well_id": f"W{well+1}",
                    "occupancy": "empty" if well == 2 else "single",
                    "species": None if well == 2 else "D. melanogaster",
                    "frames": [
                        {
                            "frame_index": f,
                            "points": points,
                            "active": True,
                            "partial": False,
                        }
                        for f in range(6)
                    ],
                }
            )
        for f in range(6):
            image = np.full((400, 600, 3), 145 if role == "train" else 175, np.uint8)
            for well in range(6):
                x = 100 + (well % 3) * 200
                y = 100 + (well // 3) * 200
                cv2.circle(image, (x, y), 65, (215, 215, 215), 3)
                if well != 2:
                    cv2.ellipse(
                        image,
                        (x + f * 2, y),
                        (25, 9),
                        30 + f * 5,
                        0,
                        360,
                        (25, 50, 100),
                        -1,
                    )
            path = f"images/frame-{f+1:06d}.png"
            cv2.imwrite(str(output / path), image)
            images.append(
                {
                    "path": path,
                    "frame_number": f + 1,
                    "frame_index": f,
                    "frame_id": f"{role}-frame{f}.png",
                    "original_filename": f"frame{f}.png",
                    "sha256": hashlib.sha256((output / path).read_bytes()).hexdigest(),
                    "width": 600,
                    "height": 400,
                }
            )
            for w in wells[:3]:
                label = (
                    "empty"
                    if w["occupancy"] == "empty"
                    else "coma" if f < 2 else "awake"
                )
                annotations.append(
                    {
                        "id": f"f{f+1}-{w['well_uid']}",
                        "frame_index": f,
                        "frame_number": f + 1,
                        "image_path": path,
                        "well_uid": w["well_uid"],
                        "well_id": w["well_id"],
                        "label": label,
                        "label_source": (
                            "manual_occupancy" if label == "empty" else "manual"
                        ),
                        "occupancy": w["occupancy"],
                        "species": w["species"],
                        "boundary_normalized": w["frames"][f]["points"],
                        "eligible_for_training": True,
                        "boundary_partial": False,
                    }
                )
        manifest = {
            "schema_version": 2,
            "task": "ccrt",
            "role": role,
            "mode": "sequence",
            "experiment_id": "synthetic-" + role,
            "interval": 2,
            "time_origin": {"confirmed": True, "offset_seconds": 10},
            "analysis_start_index": 0,
            "species": "D. melanogaster",
            "images": images,
            "annotations": annotations,
            "wells": wells,
            "coverage": {
                "detected_wells": 6,
                "expected_samples": 36,
                "annotated_samples": 18,
                "eligible_samples": 18,
            },
        }
        (output / "annotations.json").write_text(json.dumps(manifest, indent=2))
    return root / "train", root / "test"
