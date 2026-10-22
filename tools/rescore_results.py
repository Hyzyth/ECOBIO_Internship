#!/usr/bin/env python3
"""Command entry point; implementation: flyscope.reporting.rescore."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyscope.reporting.rescore import main

if __name__ == "__main__":
    main()
