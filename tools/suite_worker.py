#!/usr/bin/env python3
"""Long-lived inference worker; same interpreter as Flask."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyscope.models.runtime import SuiteModel

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
args = p.parse_args()
model = SuiteModel(args.model, args.device)
try:
    for line in sys.stdin:
        try:
            r = json.loads(line)
            started = time.perf_counter()
            content = Path(r["image"]).read_bytes()
            read_seconds = time.perf_counter() - started
            records = model.predict(
                content,
                r["wells"],
                r.get("session"),
                r.get("frame"),
            )
            print(
                json.dumps(
                    {
                        "records": records,
                        "timing": {
                            "file_read_s": read_seconds,
                            **getattr(model, "last_timing", {}),
                        },
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)
finally:
    model.close()
