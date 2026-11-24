"""Run real React/API contract tests through the frontend's owned test runner."""
from pathlib import Path
import shutil
import subprocess


def test_frontend_public_and_failure_contracts():
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    assert npm, "Node/npm is required to verify frontend behavior"
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [npm, "run", "test", "--", "--run"],
        cwd=root / "web",
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "passed" in completed.stdout, completed.stdout

