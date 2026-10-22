#!/usr/bin/env python3
"""Prepare image/keypoint projects for the app's Pose annotation page."""
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyscope.data.pose_labels import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=48)
    p.add_argument(
        "--size",
        type=int,
        default=128,
        help="Well crop resolution; match model_suite --image-size (the pose network may resize it internally)",
    )
    p.add_argument("--skeleton", help="pose.json from your existing pose model")
    p.add_argument("--wells", nargs="+")
    a = p.parse_args()
    m = prepare(a.dataset, a.output, a.count, a.size, a.skeleton, a.wells)
    print(
        json.dumps(
            {
                "examples": len(m["examples"]),
                "individuals": len({e["group_id"] for e in m["examples"]}),
                "output": a.output,
            }
        )
    )


if __name__ == "__main__":
    main()
