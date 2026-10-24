"""Public commands/assets work outside the repository working directory."""

from contextlib import chdir
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from flyscope.paths import ROOT


class EntrypointTests(unittest.TestCase):
    def test_existing_command_names_work_from_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            for command, arguments in [
                ("model_suite.py", ["plan", "--help"]),
                ("model_experiments.py", ["evaluate", "--help"]),
                ("sleap_pose.py", ["train", "--help"]),
                ("pose_diagnostics.py", ["--help"]),
                ("audit_comparison.py", ["--help"]),
                ("plot_training.py", ["--help"]),
                ("pose_labels.py", ["--help"]),
                ("rescore_results.py", ["--help"]),
                ("inspect_model.py", ["--help"]),
            ]:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools" / command), *arguments],
                    cwd=temporary,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_dataset_and_geometry_imports_do_not_initialize_model_frameworks(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from flyscope.data.datasets import Dataset; "
            "from flyscope.imaging.alignment import align_wells; "
            "assert 'flyscope.models.registry' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "assert 'keras' not in sys.modules"
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-c", code, str(ROOT)],
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_web_assets_and_standalone_export_schema_ignore_shell_directory(self):
        from app import app

        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            client = app.test_client()
            for route in (
                "/ccrt",
                "/eggs",
                "/pose",
                "/static/review.js",
                "/static/pose.js",
                "/api/dataset-format",
            ):
                with client.get(route) as response:
                    self.assertEqual(response.status_code, 200, route)
            with client.get("/static/comparaisons.json") as response:
                self.assertEqual(response.status_code, 404)
