"""App adapters for separately installed PyTorch/pose inference runtimes."""

import atexit
from collections import OrderedDict
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading

WORKERS = OrderedDict()
WORKER_LOCK = threading.RLock()


class SuiteAdapter:
    def __init__(self, path, meta):
        self.path = Path(path).resolve()
        self.meta = meta
        self.process = None
        self.stderr = None
        self.lock = threading.Lock()
        self.python = os.environ.get("FLYSCOPE_SUITE_PYTHON")
        self.available = bool(
            self.python
            and Path(self.python).is_file()
            and (
                meta["architecture"] != "sleap-rf"
                or os.environ.get("FLYSCOPE_SLEAP_PYTHON")
            )
        )
        self.metadata = {
            "name": meta["name"],
            "version": meta["version"],
            "task": "ccrt",
            "classes": meta["classes"],
            "architecture": meta["architecture"],
            "requires_sequence": meta["requires_sequence"],
            "description": "Trained CCRT model; confidence/streak recovery decisions retain raw predictions.",
            "preprocessing": meta["preprocessing"],
            "class_support": meta["class_support"],
            "recovery_policy": meta["recovery_policy"],
        }

    def close(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
            self.process = None
        if self.stderr:
            self.stderr.close()
            self.stderr = None

    def predict(self, content, wells):
        return self.predict_sequence(content, wells)

    def predict_sequence(self, content, wells, session=None, frame=None):
        if not self.available:
            raise ValueError(
                "Configure the suite and SLEAP interpreters before selecting this trained model."
            )
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.close()
                with WORKER_LOCK:
                    for identity, adapter in list(WORKERS.items()):
                        if len(WORKERS) < 2:
                            break
                        if adapter is not self and adapter.lock.acquire(blocking=False):
                            try:
                                adapter.close()
                                WORKERS.pop(identity, None)
                            finally:
                                adapter.lock.release()
                    self.stderr = tempfile.TemporaryFile(mode="w+t")
                    command = [
                        self.python,
                        str(
                            Path(__file__).resolve().parents[2]
                            / "tools/suite_worker.py"
                        ),
                        "--model",
                        str(self.path),
                        "--device",
                        os.environ.get("FLYSCOPE_INFERENCE_DEVICE", "cpu"),
                    ]
                    self.process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=self.stderr,
                        text=True,
                        bufsize=1,
                    )
                    WORKERS[str(self.path)] = self
            with tempfile.TemporaryDirectory(prefix="flyscope-inference-") as tmp:
                image = Path(tmp) / "image"
                image.write_bytes(content)
                request = {
                    "image": str(image),
                    "wells": wells,
                    "session": session,
                    "frame": frame,
                }
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                with selectors.DefaultSelector() as selector:
                    selector.register(self.process.stdout, selectors.EVENT_READ)
                    if not selector.select(180):
                        self.close()
                        raise RuntimeError(
                            "Model inference timed out; start a new analysis session."
                        )
                reply = self.process.stdout.readline()
                if not reply:
                    self.stderr.seek(0)
                    message = self.stderr.read()[-2000:]
                    self.close()
                    raise RuntimeError("Model worker failed: " + message)
                value = json.loads(reply)
                if value.get("error"):
                    raise RuntimeError(value["error"])
                return value["records"]


def registered_suite_adapters():
    directory = os.environ.get("FLYSCOPE_SUITE_MODELS_DIR")
    if not directory:
        return {}
    adapters = {}
    for path in sorted(Path(directory).rglob("model.json")):
        meta = json.loads(path.read_text())
        if meta.get("kind") != "flyscope-suite-model":
            continue
        if meta.get("architecture") not in ["resnet18", "sleap-rf", "deepethogram"]:
            raise ValueError("Unsupported suite model architecture.")
        adapters["suite-" + meta["version"][:16]] = SuiteAdapter(path, meta)
    return adapters


@atexit.register
def close_workers():
    for adapter in list(WORKERS.values()):
        adapter.close()
