#!/usr/bin/env python3
"""Long-lived inference worker; same interpreter as Flask."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.suite_models import SuiteModel

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
args = p.parse_args()
model = SuiteModel(args.model, args.device)
try:
    for line in sys.stdin:
        try:
            r = json.loads(line)
            records = model.predict(
                Path(r["image"]).read_bytes(),
                r["wells"],
                r.get("session"),
                r.get("frame"),
            )
            print(json.dumps({"records": records}, allow_nan=False), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)
finally:
    model.close()
