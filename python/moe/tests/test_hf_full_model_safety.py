from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VENV313_PYTHON = REPO_ROOT / ".venv313" / "bin" / "python"


@unittest.skipUnless(VENV313_PYTHON.exists(), ".venv313 python is required for HF safety CLI tests")
class TestFullModelSafetyCli(unittest.TestCase):
    def _run_cli(self, *args: object) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        return subprocess.run(
            [str(VENV313_PYTHON), *[str(arg) for arg in args]],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

    def test_gemma_ref_refuses_no_offload_before_load(self):
        result = self._run_cli("python/moe/gemma_ref.py", "--no-offload")
        combined = result.stdout + result.stderr

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gemma_ref: refusing to load the full HF model without disk offload", combined)
        self.assertNotIn("loading tokenizer + model", combined)

    def test_step5_helper_refuses_unsafe_cpu_budget_before_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = self._run_cli(
                "tmp/gemma_step5_amplification_report.py",
                "--max-cpu-memory-gib",
                "24",
                "--report-path",
                tmp_path / "report.txt",
                "--phase-path",
                tmp_path / "phase.json",
            )

        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gemma_step5_amplification_report: refusing CPU memory budget 24 GiB", combined)
        self.assertIn("safe ceiling is 12 GiB", combined)
        self.assertNotIn("loading HF model", combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)