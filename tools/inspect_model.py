#!/usr/bin/env python3
"""Resolution audit and ResNet feature/Grad-CAM gallery."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyscope.reporting.inspection import main

if __name__ == "__main__":
    main()
