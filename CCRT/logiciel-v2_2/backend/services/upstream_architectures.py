"""Load verified upstream DeepEthogram architectures without its obsolete trainer.

The author-distributed wheel pins Torch 2.4 and rejects Hydra (used by SLEAP-NN).
Only its self-contained Torch modules are loaded, in their own namespace. No
third-party source is redistributed in this repository. The upstream research
license applies: https://github.com/jbohnslav/deepethogram/blob/master/license.txt
"""

import hashlib
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import urllib.request
import zipfile

WHEEL_URL = "https://files.pythonhosted.org/packages/26/e9/7ffb87cb4953c56a2990518d9e9acfdb804ba319429938a7080fcd8e9362/deepethogram-0.5.0-py3-none-any.whl"
WHEEL_SHA256 = "7ab98bba863d3e09bc1321c52e9e03d07e110b357c9346229e538d2c28a9fbd4"
SOURCES = {
    "TinyMotionNet.py": (
        "deepethogram/flow_generator/models/TinyMotionNet.py",
        "4da1551f4d9819f0e40286b6bf3f2895cf5a9fc5038512371a5c2eae9c3fa910",
    ),
    "components.py": (
        "deepethogram/flow_generator/models/components.py",
        "2fa4d6a31677ed87e3946fa8f4a4a10476cca0a181792f87826f4b541ea0d7c5",
    ),
    "tgm.py": (
        "deepethogram/sequence/models/tgm.py",
        "da112b1924c9ac4fff7522124566ec861807224584ae820feee837080599fcbe",
    ),
}
LOCK = threading.Lock()


def cache_directory():
    base = os.environ.get("FLYSCOPE_CACHE_DIR")
    if not base:
        base = (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "flyscope"
        )
    return Path(base) / ("deepethogram-0.5.0-" + WHEEL_SHA256[:12])


def prepare():
    root = cache_directory()
    with LOCK:
        missing = False
        for name, (_, expected) in SOURCES.items():
            path = root / name
            if not path.is_file():
                missing = True
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    f"Cached upstream source hash differs: {path}. Remove the cache directory and prepare again."
                )
        if missing:
            with urllib.request.urlopen(WHEEL_URL, timeout=90) as response:
                content = response.read(16 * 1024 * 1024)
            if hashlib.sha256(content).hexdigest() != WHEEL_SHA256:
                raise ValueError("DeepEthogram download failed SHA-256 verification.")
            root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                for name, (source, expected) in SOURCES.items():
                    data = archive.read(source)
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError(
                            "Upstream architecture failed SHA-256 verification."
                        )
                    with tempfile.NamedTemporaryFile(dir=root, delete=False) as file:
                        file.write(data)
                        temporary = Path(file.name)
                    os.replace(temporary, root / name)
    return root


def architectures():
    root = prepare()
    namespace = "_flyscope_deepethogram_" + WHEEL_SHA256[:12]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(root)]
        sys.modules[namespace] = package
    motion = importlib.import_module(namespace + ".TinyMotionNet").TinyMotionNet
    temporal = importlib.import_module(namespace + ".tgm").TGM
    return motion, temporal
