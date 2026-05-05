"""Lint checks. Currently: every tracked .py file must be black-clean.

We run ``black --check`` against the project's Python files and fail
the test if any file would be reformatted. The test is fast because
black short-circuits when files are already formatted.
"""

import os
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _python_files():
    """Yield project .py paths, excluding generated / vendored dirs."""
    skip_dirs = {".git", "__pycache__", "build", "dist", ".venv", "venv"}
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in path.relative_to(REPO_ROOT).parts):
            continue
        yield path


class TestBlackFormatting(unittest.TestCase):
    def test_all_python_files_are_black_clean(self):
        black = shutil.which("black")
        if black is None:
            self.skipTest("black not installed")
        files = sorted(_python_files())
        self.assertGreater(len(files), 0, "no .py files discovered")
        result = subprocess.run(
            [black, "--check", "--quiet", *[str(p) for p in files]],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # ``black --check`` prints offending paths to stderr.
            offenders = result.stderr.strip() or result.stdout.strip()
            self.fail("black --check failed; run `black .` to fix:\n" + offenders)


if __name__ == "__main__":
    unittest.main()
