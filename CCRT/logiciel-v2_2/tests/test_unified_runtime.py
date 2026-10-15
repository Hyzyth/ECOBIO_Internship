import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.services import upstream_architectures as upstream
from tools.sleap_pose import read_meta


class UnifiedRuntimeTests(unittest.TestCase):
    def test_modified_cached_architecture_is_rejected_before_import_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "TinyMotionNet.py").write_text("unexpected source")
            with patch.object(
                upstream, "cache_directory", return_value=root
            ), patch.object(upstream.urllib.request, "urlopen") as fetch:
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    upstream.prepare()
                fetch.assert_not_called()

    def test_legacy_pose_checkpoint_has_an_explicit_migration_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pose.json").write_text(json.dumps({"configuration": {}}))
            with self.assertRaisesRegex(ValueError, "legacy TensorFlow.*Retrain"):
                read_meta(root)

    def test_gpu_required_check_does_not_silently_fall_back_to_cpu(self):
        from tools.check_runtime import check

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                check(require_cuda=True)
