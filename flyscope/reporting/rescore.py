#!/usr/bin/env python3
"""Re-score saved predictions for CCRT timing without weights or another training."""
import argparse
import json
from pathlib import Path
from .metrics import metrics, write_comparison
import hashlib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tolerances", type=int, nargs="+", default=[1, 5, 7, 8, 10, 30, 60])
    args = p.parse_args()
    if any(t < 0 for t in args.tolerances):
        p.error("Tolerances must be nonnegative integer frame counts.")
    out = Path(args.output)
    if out.exists():
        p.error("Use a new output directory; original results are preserved.")
    out.mkdir(parents=True)
    compact = []
    sources = []
    for file in sorted(Path(args.results).rglob("*.json")):
        if not file.name.startswith(("resnet18-", "deepethogram-", "sleap-rf-")):
            continue
        raw = file.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        r = json.loads(raw)
        del raw
        if "predictions" not in r or "model" not in r:
            continue
        sources.append({"path": str(file.resolve()), "sha256": source_hash})
        timing = r.get("test_timing") or {
            "mode": "sequence" if r["metrics"]["recovery"]["wells"] else "single",
            "time_origin": {"confirmed": False},
        }
        r["metrics"] = metrics(r.pop("predictions"), timing, args.tolerances)
        compact.append(r)
        print(
            r["model"]["architecture"],
            r["model"]["selection"]["id"],
            r["metrics"]["recovery"]["mae_frames"],
            flush=True,
        )
    if not compact:
        p.error("No saved per-model predictions found.")
    write_comparison(compact, out)
    (out / "sources.json").write_text(json.dumps(sources, indent=2))
    (out / "recovery-audit.json").write_text(
        json.dumps(
            [
                {
                    "model": r["model"]["architecture"],
                    "run": r["model"]["selection"]["id"],
                    "recovery": r["metrics"]["recovery"],
                }
                for r in compact
            ],
            indent=2,
        )
    )
    (out / "README.txt").write_text(
        "CCRT scoring v2. No predictions, weights or policies were changed. Old results lacking test_timing are scored in frames only. MAE is conditional on predictions; tolerance success counts every exact human recovery, including missing estimates as failures. No retraining required.\n"
    )


if __name__ == "__main__":
    main()
