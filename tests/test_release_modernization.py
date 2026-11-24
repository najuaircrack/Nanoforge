from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from nanoforge.config import load_config
from nanoforge.model.transformer import build_model

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_public_model_presets_load_and_moe_is_real():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        if path.name == "project-example.yaml":
            continue
        cfg = load_config(path)
        assert cfg.training.batch_size == cfg.training.micro_batch_size * cfg.training.grad_accum_steps
        assert cfg.data.seq_len <= cfg.model.max_seq_len
    cfg = load_config(ROOT / "configs" / "moe-large.yaml")
    assert cfg.model.ffn_type == "moe"
    # Use the preset's actual MoE settings with a small CPU construction.
    cfg.model.d_model = 32
    cfg.model.n_layers = 1
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.vocab_size = 260
    cfg.model.max_seq_len = 64
    model = build_model(cfg)
    assert any(type(module).__name__ == "MoEFeedForward" for module in model.modules())


def test_public_fixture_writer_never_overwrites_and_has_separate_groups(tmp_path):
    module = _script("make_public_fixtures")
    out = tmp_path / "public"
    paths = module.write_fixtures(out, kind="c")
    assert len(paths) >= 8
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert all(record["license"] == "CC0-1.0" and record["consent"] is True for record in records)
    assert all(record.get("code") for record in records)
    with pytest.raises(FileExistsError):
        module.write_fixtures(out, kind="c")


def test_synthetic_chat_fixture_and_config_match(tmp_path):
    paths = _script("make_public_fixtures").write_fixtures(tmp_path / "chat", kind="chat")
    assert all(json.loads(path.read_text(encoding="utf-8"))["messages"] for path in paths)
    cfg = load_config(ROOT / "configs" / "nano-chat-smoke.yaml")
    assert cfg.model.vocab_size == 260
    assert cfg.data.tokenizer_type == "byte"
    assert cfg.data.seq_len == cfg.model.max_seq_len
    assert cfg.training.resume_from_checkpoint is None


def test_public_sft_template_has_disjoint_heldout_prompts(tmp_path):
    from nanoforge.training.pretrained import PretrainedAdaptationConfig

    template = json.loads((ROOT / "configs" / "pretrained-lora-example.json").read_text(encoding="utf-8"))
    cfg = PretrainedAdaptationConfig(**template)
    assert not cfg.allow_network and cfg.validation_path
    train, val = _script("make_public_fixtures").write_fixtures(tmp_path / "sft", kind="sft")
    def records(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {row["prompt"] for row in records(train)}.isdisjoint({row["prompt"] for row in records(val)})
    assert all(row["response"] for row in records(train) + records(val))


def test_release_dependency_contract():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert any(value.startswith("cloudpickle") for value in project["dependencies"])
    assert "torch>=2.6" in project["dependencies"]
    extras = project["optional-dependencies"]
    expected = {"pretrained": "peft", "adaptation": "transformers", "qlora": "bitsandbytes",
                "serve": "python-multipart", "export": "onnxscript", "dev": "httpx"}
    for extra, dependency in expected.items():
        assert any(value.startswith(dependency) for value in extras[extra])
    assert any(value.startswith("setuptools>=77") for value in extras["dev"])


def test_docker_source_allowlist_excludes_private_context():
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    assert not any(line.strip().upper() == "COPY . ." for line in lines)
    copied = [line for line in lines if line.startswith("COPY ")]
    assert copied and all(".env" not in line and "data/" not in line for line in copied)
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "**" in ignore
    assert "!src/" in ignore and "!src/**" in ignore
    assert not any(line.startswith("!") and any(part in line for part in
               [".env", ".agents", ".codex", ".unlazy", "data/", "runs/"]) for line in ignore)


def test_config_docs_generated_from_actual_schema(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_config_docs.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cli = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_cli_docs.py"), "--check"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert cli.returncode == 0, cli.stdout + cli.stderr


def test_portable_smoke_executes_public_and_failure_paths(tmp_path):
    runner = _script("run_smoke")
    out = tmp_path / "smoke"
    result = runner.run_smoke(out, steps=2)
    assert result["train_tokens"] > 0 and result["val_tokens"] > 0
    assert result["checkpoint"].is_file()
    assert result["eval_loss"] >= 0
    with pytest.raises(FileExistsError):
        runner.run_smoke(out, steps=2)


def test_ci_contains_windows_web_native_package_and_optional_jobs():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    jobs = ci["jobs"]
    for name in ("python", "web", "native", "package", "container"):
        assert name in jobs
    matrix = jobs["python"]["strategy"]["matrix"]
    assert "windows-latest" in matrix["os"]
    install = "\n".join(str(step.get("run", "")) for step in jobs["python"]["steps"])
    assert "npm ci" in install and "serve" in install and "pretrained" in install


def test_public_fixture_is_available_in_source_archive(tmp_path):
    # An sdist must carry the authored examples required by public commands.
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--no-isolation", "--outdir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archive = next(tmp_path.glob("*.tar.gz"))
    with tarfile.open(archive) as handle:
        names = handle.getnames()
    assert any(name.endswith("/scripts/make_public_fixtures.py") for name in names)
    assert any(name.endswith("/configs/c-tiny.yaml") for name in names)
    paths = [name.split("/", 1)[1] for name in names if "/" in name]
    assert not any(name.startswith(("data/", "runs/", "models/", ".unlazy/", ".latticekin/"))
                   or name.startswith(".env") for name in paths)


def test_wheel_has_public_cli_and_ui_without_private_state(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as handle:
        names = handle.namelist()
        entry_points = handle.read(next(name for name in names if name.endswith("entry_points.txt"))).decode()
    assert "latticekin = nanoforge.cli:main" in entry_points
    assert "nanoforge = nanoforge.cli:main" in entry_points
    assert "nanoforge/web_dist/index.html" in names
    assert not any(name.startswith(("data/", "runs/", ".unlazy/")) or ".env" in name for name in names)
    # Install the actual wheel without consulting an index or the source PYTHONPATH.
    environment = tmp_path / "installed"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
                   check=True, capture_output=True, text=True, timeout=90)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(wheel)],
                   check=True, capture_output=True, text=True, timeout=90)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    check = subprocess.run(
        [str(python), "-c", "from pathlib import Path; import sys, nanoforge; assert Path(nanoforge.__file__).resolve().is_relative_to(Path(sys.prefix))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    help_result = subprocess.run([str(python), "-m", "nanoforge", "--help"], cwd=tmp_path,
                                 env=env, capture_output=True, text=True, timeout=60)
    assert help_result.returncode == 0 and "latticekin" in help_result.stdout

