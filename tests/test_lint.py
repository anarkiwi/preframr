"""Lint checks: black formatting, no narrative comments, short docstrings."""

import ast
import io
import os
import re
import shutil
import subprocess
import tokenize
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LINT_DIRS = ("preframr", "tests")
ALLOWED_DIRECTIVE_RE = re.compile(
    r"^(" r"pylint:|" r"noqa(\s|:|$)|" r"type:\s*ignore|" r"fmt:\s*(on|off)" r")"
)
MAX_DOCSTRING_LINES = 5


def _python_files():
    skip_dirs = {".git", "__pycache__", "build", "dist", ".venv", "venv"}
    for top in LINT_DIRS:
        for path in (REPO_ROOT / top).rglob("*.py"):
            if any(part in skip_dirs for part in path.relative_to(REPO_ROOT).parts):
                continue
            yield path


def _is_directive(comment: str) -> bool:
    body = comment.lstrip("#").strip()
    return bool(body) and bool(ALLOWED_DIRECTIVE_RE.match(body))


def _is_shebang(comment: str, lineno: int) -> bool:
    return lineno == 1 and comment.startswith("#!")


class TestBlackFormatting(unittest.TestCase):
    def test_all_python_files_are_black_clean(self):
        black = shutil.which("black")
        if black is None:
            self.skipTest("black not installed")
        files = sorted(_python_files())
        self.assertGreater(len(files), 0, "no .py files discovered")
        subprocess_env = {
            k: v for k, v in os.environ.items() if not k.startswith("COV_CORE_")
        }
        result = subprocess.run(
            [
                black,
                "--check",
                "--quiet",
                "--workers",
                "1",
                *[str(p) for p in files],
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=subprocess_env,
        )
        if result.returncode != 0:
            offenders = result.stderr.strip() or result.stdout.strip()
            self.fail("black --check failed; run `black .` to fix:\n" + offenders)


class TestNoNarrativeComments(unittest.TestCase):
    """Every `# ...` line must be a tooling directive or a line-1 shebang."""

    def test_no_narrative_comments(self):
        offenders = []
        for path in sorted(_python_files()):
            try:
                src = path.read_bytes()
            except OSError:
                continue
            try:
                tokens = list(tokenize.tokenize(io.BytesIO(src).readline))
            except (tokenize.TokenError, IndentationError):
                continue
            for tok in tokens:
                if tok.type != tokenize.COMMENT:
                    continue
                lineno = tok.start[0]
                if _is_shebang(tok.string, lineno):
                    continue
                if _is_directive(tok.string):
                    continue
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {tok.string.strip()}")
        if offenders:
            head = offenders[:20]
            more = (
                f"\n  ... and {len(offenders) - 20} more" if len(offenders) > 20 else ""
            )
            self.fail(
                "Non-directive comments found "
                f"(allowed: pylint: / noqa / type: ignore / fmt: on/off / "
                f"line-1 shebang):\n  " + "\n  ".join(head) + more
            )


class TestDocstringShape(unittest.TestCase):
    """Module/class/function docstrings must be one short paragraph (≤5 lines)."""

    def test_docstrings_one_short_paragraph(self):
        import inspect

        offenders = []
        for path in sorted(_python_files()):
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    continue
                raw = ast.get_docstring(node, clean=False)
                if not raw:
                    continue
                text = inspect.cleandoc(raw).strip()
                if not text:
                    continue
                lines = text.splitlines()
                rel = path.relative_to(REPO_ROOT)
                lineno = getattr(node, "lineno", 0)
                if any(not ln.strip() for ln in lines):
                    offenders.append(
                        f"{rel}:{lineno}: docstring has blank-line paragraph break"
                    )
                    continue
                if len(lines) > MAX_DOCSTRING_LINES:
                    offenders.append(
                        f"{rel}:{lineno}: docstring is {len(lines)} lines "
                        f"(max {MAX_DOCSTRING_LINES})"
                    )
        if offenders:
            head = offenders[:20]
            more = (
                f"\n  ... and {len(offenders) - 20} more" if len(offenders) > 20 else ""
            )
            self.fail(
                "Docstrings must be one short paragraph "
                f"(≤{MAX_DOCSTRING_LINES} lines, no blank-line breaks):\n  "
                + "\n  ".join(head)
                + more
            )


if __name__ == "__main__":
    unittest.main()
