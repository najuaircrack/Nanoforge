"""Portable offline checks; install Python/web/native dependencies beforehand."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, cwd: Path = ROOT, env: dict | None = None) -> None:
    print("CHECK:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", action="store_true", help="Also run web test/build/lint.")
    parser.add_argument("--native", action="store_true", help="Also run locked Rust tests.")
    parser.add_argument("--package", action="store_true", help="Also build wheel and source archive.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--lint-only", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    run([sys.executable, "-m", "ruff", "check", "."], env=env)
    run([sys.executable, "scripts/generate_config_docs.py", "--check"], env=env)
    run([sys.executable, "scripts/generate_cli_docs.py", "--check"], env=env)
    if args.lint_only:
        return
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm or not (ROOT / "web" / "node_modules").is_dir():
        raise SystemExit("Python workflow tests require Node/npm and web dependencies. Run npm ci in web before offline checks.")
    run([sys.executable, "-m", "pytest", "-q"], env=env)
    if args.web or args.all:
        run([npm, "test", "--", "--run"], cwd=ROOT / "web", env=env)
        run([npm, "run", "build"], cwd=ROOT / "web", env=env)
        run([npm, "run", "lint"], cwd=ROOT / "web", env=env)
    if args.native or args.all:
        cargo = shutil.which("cargo")
        if not cargo:
            raise SystemExit("Rust checks require cargo on PATH.")
        native_env = dict(env)
        if os.name == "nt":
            native_env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + native_env.get("PATH", "")
        run([cargo, "test", "--locked", "--offline"], cwd=ROOT / "native" / "nanoforge-tokenizers", env=native_env)
    if args.package or args.all:
        run([sys.executable, "-m", "build", "--no-isolation"], env=env)
    print("All requested offline checks passed", flush=True)


if __name__ == "__main__":
    main()

