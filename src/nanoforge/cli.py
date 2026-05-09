from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _paths(values: list[str]) -> list[Path]:
    return [Path(v) for v in values]


def cmd_params(args: argparse.Namespace) -> None:
    from nanoforge.config import load_config
    from nanoforge.model.transformer import NanoforgeForCausalLM

    cfg = load_config(args.config)
    model = NanoforgeForCausalLM(cfg.model)
    print(f"parameters: {model.estimate_num_params():,}")
    print(f"non_embedding_parameters: {model.estimate_num_params(non_embedding=True):,}")


def cmd_registries(args: argparse.Namespace) -> None:
    import json

    from nanoforge.registry import registry_snapshot

    snapshot = registry_snapshot()
    if args.name:
        snapshot = {args.name: snapshot[args.name]}
    print(json.dumps(snapshot, indent=2, sort_keys=True))


def cmd_validate_config(args: argparse.Namespace) -> None:
    from nanoforge.config import load_config

    cfg = load_config(args.config)
    print("valid: true")
    print(f"block: {cfg.model.block_type}")
    print(f"attention: {cfg.model.attention_backend}")
    print(f"ffn: {cfg.model.ffn_type}")
    print(f"normalization: {cfg.model.normalization}")
    print(f"tokenizer: {cfg.data.tokenizer_type}")


def cmd_train_tokenizer(args: argparse.Namespace) -> None:
    from dataclasses import asdict
    import json

    from nanoforge.data.tokenizer import (
        train_bpe_tokenizer,
        train_native_bpe_tokenizer,
        train_python_bpe_tokenizer,
        train_sentencepiece_tokenizer,
        train_wordpiece_tokenizer,
    )

    files: list[Path] = []
    for root in _paths(args.input):
        files.extend([p for p in root.rglob("*") if p.is_file()] if root.is_dir() else [root])
    columns = tuple(args.text_column or ())
    if args.type == "bpe":
        report = train_bpe_tokenizer(
            files,
            args.out,
            args.vocab_size,
            args.min_frequency,
            text_key=args.text_key,
            text_columns=columns,
            dry_run=args.dry_run,
            max_records=args.max_records,
        )
    elif args.type == "python-bpe":
        report = train_python_bpe_tokenizer(
            files,
            args.out,
            args.vocab_size,
            args.min_frequency,
            text_key=args.text_key,
            text_columns=columns,
            dry_run=args.dry_run,
            max_records=args.max_records,
        )
    elif args.type == "native-bpe":
        report = train_native_bpe_tokenizer(
            files,
            args.out,
            args.vocab_size,
            args.min_frequency,
            text_key=args.text_key,
            text_columns=columns,
            dry_run=args.dry_run,
            max_records=args.max_records,
            show_progress=not args.no_progress
        )
    elif args.type == "wordpiece":
        report = train_wordpiece_tokenizer(
            files,
            args.out,
            args.vocab_size,
            args.min_frequency,
            text_key=args.text_key,
            text_columns=columns,
            dry_run=args.dry_run,
            max_records=args.max_records,
        )
    else:
        model_type = "unigram" if args.type == "unigram" else "bpe"
        report = train_sentencepiece_tokenizer(
            files,
            args.out,
            args.vocab_size,
            model_type=model_type,
            text_key=args.text_key,
            text_columns=columns,
            dry_run=args.dry_run,
            max_records=args.max_records,
        )
    print(json.dumps(asdict(report), indent=2))


def cmd_prepare(args: argparse.Namespace) -> None:
    from nanoforge.data.dataset import build_packed_dataset
    from nanoforge.data.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_path)
    progress = None
    if not args.no_progress:
        from tqdm import tqdm

        progress = tqdm(desc="prepare", unit="docs", dynamic_ncols=True)

        def update_bar(stats):
            progress.update(max(0, stats.records_seen - progress.n))
            progress.set_postfix(
                train=f"{stats.train_tokens:,}",
                val=f"{stats.val_tokens:,}",
                shards=stats.shards,
            )

    else:
        update_bar = None
    build_packed_dataset(
        _paths(args.input),
        args.out,
        tokenizer,
        val_fraction=args.val_fraction,
        code_only=args.code_only,
        jsonl=args.jsonl,
        jsonl_text_key=args.jsonl_text_key,
        min_chars=args.min_chars,
        mode=args.mode,
        loss_masking=args.loss_masking,
        tokenizer_batch_size=args.tokenizer_batch_size,
        progress_callback=update_bar,
        text_columns=tuple(args.text_column or ()),
        seq_len=args.seq_len,
    )
    if progress is not None:
        progress.close()


def cmd_inspect_dataset(args: argparse.Namespace) -> None:
    from nanoforge.data.formats import inspect_dataset

    stats = inspect_dataset(_paths(args.input), text_key=args.text_key, limit=args.limit)
    print(f"records: {stats.records}")
    print(f"bytes_read: {stats.bytes_read}")
    print(f"formats: {stats.formats}")
    print(f"fields: {stats.fields}")
    print(f"schemas: {stats.schemas}")
    print(f"text_columns: {stats.text_columns}")
    print(f"skipped_records: {stats.skipped_records}")
    print(f"invalid_records: {stats.invalid_records}")
    print(f"fingerprints: {stats.fingerprints}")
    if stats.issues:
        print("issues:")
        for issue in stats.issues[:20]:
            print(f"- {issue.kind}: {issue.source}: {issue.message}")


def cmd_validate_dataset(args: argparse.Namespace) -> None:
    from nanoforge.data.formats import inspect_dataset

    stats = inspect_dataset(_paths(args.input), text_key=args.text_key, limit=args.limit)
    has_records = stats.records > 0
    has_errors = any(issue.kind == "error" for issue in stats.issues)
    print(f"valid: {has_records and not has_errors}")
    print(f"records_checked: {stats.records}")
    print(f"skipped_records: {stats.skipped_records}")
    print(f"invalid_records: {stats.invalid_records}")
    if stats.issues:
        print("issues:")
        for issue in stats.issues[:50]:
            print(f"- {issue.kind}: {issue.source}: {issue.message}")
    if not has_records or has_errors:
        raise SystemExit(1)


def cmd_clean_dataset(args: argparse.Namespace) -> None:
    import json

    from nanoforge.data.cleaning import CleaningConfig, clean_records
    from nanoforge.data.formats import iter_dataset_records

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = CleaningConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        collapse_whitespace=args.collapse_whitespace,
        deduplicate=not args.no_deduplicate,
        near_deduplicate=args.near_deduplicate,
        language=args.language,
    )
    records = iter_dataset_records(_paths(args.input), text_key=args.text_key, text_columns=args.text_column)
    written = 0
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for record in clean_records(records, config):
            fh.write(json.dumps({"text": record.text, "source": record.source, "metadata": record.metadata}, ensure_ascii=False))
            fh.write("\n")
            written += 1
    print(f"written: {written}")
    print(f"out: {out}")


def cmd_convert_dataset(args: argparse.Namespace) -> None:
    import json

    from nanoforge.data.formats import iter_dataset_records

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = iter_dataset_records(_paths(args.input), text_key=args.text_key, text_columns=args.text_column)
    written = 0
    if args.format == "txt":
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(record.text)
                if not record.text.endswith("\n"):
                    fh.write("\n")
                written += 1
    else:
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(json.dumps({"text": record.text, "source": record.source, "metadata": record.metadata}, ensure_ascii=False))
                fh.write("\n")
                written += 1
    print(f"written: {written}")
    print(f"out: {out}")


def cmd_tokenizer_report(args: argparse.Namespace) -> None:
    from dataclasses import asdict
    import json

    from nanoforge.data.tokenizer import load_tokenizer
    from nanoforge.data.tokenizer_metrics import evaluate_tokenizer, save_tokenizer_report

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_path)
    report = evaluate_tokenizer(tokenizer, _paths(args.input), text_key=args.text_key, limit=args.limit)
    if args.out:
        save_tokenizer_report(report, args.out)
    print(json.dumps(asdict(report), indent=2))


def cmd_tokenizer_status(args: argparse.Namespace) -> None:
    from dataclasses import asdict
    import json

    from nanoforge.data.native_tokenizer import native_tokenizer_status

    _ = args
    print(json.dumps(asdict(native_tokenizer_status()), indent=2))


def cmd_benchmark_tokenizer(args: argparse.Namespace) -> None:
    from dataclasses import asdict
    import json

    from nanoforge.data.tokenizer import load_tokenizer
    from nanoforge.data.tokenizer_benchmark import benchmark_tokenizer

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_path)
    result = benchmark_tokenizer(
        tokenizer,
        _paths(args.input),
        text_key=args.text_key,
        text_columns=args.text_column,
        limit=args.limit,
        batch_size=args.batch_size,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )
    print(json.dumps(asdict(result), indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    if args.dashboard:
        import threading
        import webbrowser

        from nanoforge.config import load_config
        from nanoforge.dashboard import serve_dashboard
        from nanoforge.progress import reset_metric_file

        cfg = load_config(args.config)
        reset_metric_file(Path(cfg.training.output_dir) / "metrics.jsonl", backup=True)
        url = f"http://{args.dashboard_host}:{args.dashboard_port}"
        thread = threading.Thread(
            target=serve_dashboard,
            args=(cfg.training.output_dir, args.dashboard_host, args.dashboard_port),
            daemon=True,
        )
        thread.start()
        print(f"dashboard: {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    from nanoforge.training.trainer import train_from_config

    train_from_config(args.config)


def cmd_dashboard(args: argparse.Namespace) -> None:
    from nanoforge.dashboard import serve_dashboard

    serve_dashboard(args.run, args.host, args.port)


def _sampling_from_args(args: argparse.Namespace):
    from nanoforge.generation.sampling import SamplingConfig

    return SamplingConfig(
        mode=args.mode,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        deterministic=args.deterministic,
        stop_on_repetition=not args.no_repetition_stop,
        repetition_window=args.repetition_window,
        repetition_threshold=args.repetition_threshold,
        mirostat=args.mirostat,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    engine = _load_generation_engine(args)
    if args.beams > 1:
        if not hasattr(engine, "beam_search"):
            raise SystemExit("Beam search is only available for native Nanoforge checkpoints.")
        print(engine.beam_search(args.prompt, args.max_new_tokens, args.beams))
    else:
        print(engine.complete(args.prompt, args.max_new_tokens, _sampling_from_args(args), args.stop_token))


def cmd_chat(args: argparse.Namespace) -> None:
    engine = _load_generation_engine(args)
    print("Nanoforge chat. Press Ctrl+C or submit an empty prompt to exit.")
    try:
        while True:
            prompt = input("\nuser> ").strip()
            if not prompt:
                return
            print("assistant> ", end="", flush=True)
            for chunk in engine.stream(prompt, args.max_new_tokens, _sampling_from_args(args), args.stop_token):
                print(chunk, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print()


def _load_generation_engine(args: argparse.Namespace):
    if getattr(args, "model", None):
        from nanoforge.imports import load_imported_engine

        return load_imported_engine(args.model, device=args.device)
    if not getattr(args, "checkpoint", None):
        raise SystemExit("Provide --checkpoint for a Nanoforge checkpoint or --model for an imported model.")
    from nanoforge.generation.engine import GenerationEngine

    return GenerationEngine.from_checkpoint(args.checkpoint, device=args.device)


def cmd_import_model(args: argparse.Namespace) -> None:
    from nanoforge.imports import import_model

    entry = import_model(args.model, args.name, tokenizer=args.tokenizer, backend=args.backend)
    print(f"imported: {entry.name}")
    print(f"backend: {entry.backend}")
    print(f"format: {entry.format}")
    print(f"source: {entry.source}")


def cmd_new_config(args: argparse.Namespace) -> None:
    from nanoforge.templates import interactive_new_config

    path = interactive_new_config(args.out)
    print(f"wrote: {path}")


def cmd_auto_train(args: argparse.Namespace) -> None:
    from nanoforge.data.dataset import build_packed_dataset
    from nanoforge.data.formats import inspect_dataset
    from nanoforge.data.tokenizer import (
        load_tokenizer,
        train_bpe_tokenizer,
        train_native_bpe_tokenizer,
        train_python_bpe_tokenizer,
        train_sentencepiece_tokenizer,
        train_wordpiece_tokenizer,
    )
    from nanoforge.templates import build_cpu_config

    input_paths = _paths(args.input)
    text_columns = tuple(args.text_column or ())
    mode = args.mode
    if mode == "auto":
        inspected = inspect_dataset(input_paths, text_key=args.text_key, limit=args.inspect_limit)
        fields = {field.lower() for field in inspected.fields}
        if {"messages", "conversations"} & fields or any("messages" in cols for cols in inspected.text_columns.values()):
            mode = "chat"
        elif {"instruction", "output", "response"} <= fields or ("instruction" in fields and {"output", "response"} & fields):
            mode = "instruct"
        elif "code" in fields:
            mode = "code"
        else:
            mode = "generative"
        if not text_columns and mode == "chat":
            text_columns = ("messages",)
    loss_masking = args.loss_masking
    if loss_masking == "auto":
        loss_masking = "assistant_only" if mode == "chat" else "completion_only" if mode == "instruct" else "none"

    name = args.name
    default_tokenizer_path = f"data/tokenizers/{name}.model" if args.tokenizer in {"sentencepiece", "unigram"} else f"data/tokenizers/{name}-bpe.json"
    tokenizer_path = Path(args.tokenizer_path or default_tokenizer_path)
    packed_dir = Path(args.packed_dir or f"data/packed/{name}")
    config_path = Path(args.config_out or f"configs/{name}.yaml")
    files: list[Path] = []
    for root in input_paths:
        files.extend([p for p in root.rglob("*") if p.is_file()] if root.is_dir() else [root])

    if args.tokenizer in {"bpe", "native-bpe", "python-bpe", "wordpiece", "sentencepiece", "unigram"}:
        print(f"[1/4] training tokenizer: {args.tokenizer} -> {tokenizer_path}")
        if args.tokenizer == "native-bpe":
            train_native_bpe_tokenizer(
                files,
                tokenizer_path,
                args.vocab_size,
                args.min_frequency,
                text_key=args.text_key,
                text_columns=text_columns,
                max_records=args.max_records,
                show_progress=not args.no_progress,
            )
        elif args.tokenizer == "python-bpe":
            train_python_bpe_tokenizer(
                files,
                tokenizer_path,
                args.vocab_size,
                args.min_frequency,
                text_key=args.text_key,
                text_columns=text_columns,
                max_records=args.max_records,
            )
        elif args.tokenizer == "bpe":
            train_bpe_tokenizer(
                files,
                tokenizer_path,
                args.vocab_size,
                args.min_frequency,
                text_key=args.text_key,
                text_columns=text_columns,
                max_records=args.max_records,
            )
        elif args.tokenizer == "wordpiece":
            train_wordpiece_tokenizer(
                files,
                tokenizer_path,
                args.vocab_size,
                args.min_frequency,
                text_key=args.text_key,
                text_columns=text_columns,
                max_records=args.max_records,
            )
        else:
            train_sentencepiece_tokenizer(
                files,
                tokenizer_path.with_suffix(""),
                args.vocab_size,
                model_type="unigram" if args.tokenizer == "unigram" else "bpe",
                text_key=args.text_key,
                text_columns=text_columns,
                max_records=args.max_records,
            )
    else:
        print("[1/4] byte tokenizer selected; no tokenizer training needed")

    print(f"[2/4] preparing data: mode={mode}, loss_masking={loss_masking} -> {packed_dir}")
    tokenizer = load_tokenizer(args.tokenizer, tokenizer_path if args.tokenizer not in {"byte", "byte-native"} else None)
    progress = None
   
    from tqdm import tqdm

    progress = tqdm(desc="prepare", unit="docs", dynamic_ncols=True)

    def update_bar(stats):
        progress.update(max(0, stats.records_seen - progress.n))
        progress.set_postfix(
            train=f"{stats.train_tokens:,}",
            val=f"{stats.val_tokens:,}",
            shards=stats.shards,
        )


    build_packed_dataset(
        input_paths,
        packed_dir,
        tokenizer,
        val_fraction=args.val_fraction,
        jsonl_text_key=args.text_key,
        min_chars=args.min_chars,
        mode=mode,
        loss_masking=loss_masking,
        progress_callback=update_bar,
        tokenizer_batch_size=args.tokenizer_batch_size,
        text_columns=text_columns,
        seq_len=args.seq_len,
    )

    if progress is not None:
        progress.close()

    print(f"[3/4] writing config: {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_text = build_cpu_config(
        name=name,
        mode=mode,
        ram=args.ram,
        speed=args.speed,
        data_format="parquet chat" if mode == "chat" else "plain text",
        tokenizer_type=args.tokenizer,
        tokenizer_path="null" if args.tokenizer in {"byte", "byte-native"} else str(tokenizer_path).replace("\\", "/"),
        packed_dir=str(packed_dir).replace("\\", "/"),
        vocab_size=args.vocab_size if args.tokenizer not in {"byte", "byte-native"} else tokenizer.vocab_size,
        max_steps=args.max_steps,
        seq_len_override=args.seq_len,
        loss_masking=loss_masking,
    )
    config_path.write_text(config_text, encoding="utf-8")

    if args.no_train:
        print("[4/4] skipped training (--no-train)")
        return
    print("[4/4] training")
    from nanoforge.training.trainer import train_from_config

    train_from_config(config_path)


def cmd_serve(args: argparse.Namespace) -> None:
    from nanoforge.server import serve

    serve(args.checkpoint, args.host, args.port, args.device)


def cmd_export(args: argparse.Namespace) -> None:
    from nanoforge.export.gguf import write_gguf_manifest
    from nanoforge.export.onnx import export_onnx
    from nanoforge.generation.engine import GenerationEngine
    from nanoforge.training.checkpoint import load_checkpoint

    if args.format == "onnx":
        engine = GenerationEngine.from_checkpoint(args.checkpoint, device=args.device)
        export_onnx(engine.model, args.out)
    else:
        payload = load_checkpoint(args.checkpoint, map_location="cpu")
        write_gguf_manifest(payload["config"], args.checkpoint, args.out)


def cmd_benchmark(args: argparse.Namespace) -> None:
    from nanoforge.benchmark import benchmark_forward

    result = benchmark_forward(args.config, args.batch_size, args.steps, args.device)
    for key, value in result.items():
        print(f"{key}: {value:,.3f}" if isinstance(value, float) else f"{key}: {value}")


def cmd_profile_config(args: argparse.Namespace) -> None:
    from dataclasses import asdict
    import json

    from nanoforge.config import load_config
    from nanoforge.profiling import estimate_model_profile

    cfg = load_config(args.config)
    profile = estimate_model_profile(
        cfg.model,
        batch_size=args.batch_size,
        seq_len=args.seq_len or cfg.data.seq_len,
        bytes_per_param=args.bytes_per_param,
    )
    print(json.dumps(asdict(profile), indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from nanoforge.evaluation.metrics import evaluate_checkpoint

    result = evaluate_checkpoint(args.checkpoint, args.data, args.seq_len, args.batches, args.device)
    for key, value in result.items():
        print(f"{key}: {value:,.4f}" if isinstance(value, float) else f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanoforge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("params", help="Estimate model parameter count.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_params)

    p = sub.add_parser("registries", help="List registered Nanoforge component keys.")
    p.add_argument(
        "--name",
        choices=[
            "attention",
            "ffn",
            "activation",
            "position",
            "optimizer",
            "scheduler",
            "tokenizer",
            "sampler",
            "normalization",
            "quantization",
            "block",
        ],
    )
    p.set_defaults(func=cmd_registries)

    p = sub.add_parser("validate-config", help="Load a config and validate registry-backed keys.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_validate_config)

    p = sub.add_parser("new-config", help="Interactively create a CPU-friendly training config.")
    p.add_argument("--out", default="configs/my-model.yaml")
    p.set_defaults(func=cmd_new_config)

    p = sub.add_parser("auto-train", help="Train tokenizer, prepare data, write config, and train in one command.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument(
        "--mode",
        choices=["auto", "generative", "chat", "instruct", "completion", "code", "reasoning"],
        default="auto",
    )
    p.add_argument(
        "--loss-masking",
        choices=["auto", "none", "assistant-only", "assistant_only", "completion-only", "completion_only", "partial"],
        default="auto",
    )
    p.add_argument(
        "--tokenizer",
        choices=["byte", "byte-native", "bpe", "native-bpe", "python-bpe", "wordpiece", "sentencepiece", "unigram"],
        default="native-bpe",
    )
    p.add_argument("--tokenizer-path")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--text-key", default="text")
    p.add_argument("--text-column", action="append")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--tokenizer-batch-size", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--min-chars", type=int, default=16)
    p.add_argument("--max-records", type=int)
    p.add_argument("--inspect-limit", type=int, default=1000)
    p.add_argument("--ram", default="8GB", choices=["4GB", "8GB", "16GB", "32GB+"])
    p.add_argument("--speed", default="fast/small", choices=["fast/small", "balanced", "slow/large"])
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--packed-dir")
    p.add_argument("--config-out")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-train", action="store_true", help="Run tokenizer/config/data prep only.")
    p.set_defaults(func=cmd_auto_train)

    p = sub.add_parser("import", help="Register an external GGUF, ONNX, SafeTensors, or HuggingFace model.")
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--tokenizer", help="Optional tokenizer path/name for ONNX or custom imports.")
    p.add_argument("--backend", choices=["llama_cpp", "transformers", "onnxruntime", "safetensors"])
    p.set_defaults(func=cmd_import_model)

    p = sub.add_parser("train-tokenizer", help="Train a BPE, WordPiece, or SentencePiece tokenizer.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--type", choices=["bpe", "native-bpe", "python-bpe", "wordpiece", "sentencepiece", "unigram"], default="bpe")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--text-key", default="text")
    p.add_argument("--text-column", action="append", help="Structured text column to train on. Can be repeated.")
    p.add_argument("--max-records", type=int)
    p.add_argument("--dry-run", action="store_true", help="Scan and report training corpus health without fitting.")
    p.add_argument("--no-progress", action="store_true", help="Disable the merge progress bar.")  # ← ADD THIS
    p.set_defaults(func=cmd_train_tokenizer)

    p = sub.add_parser("prepare", help="Pack raw text/code/JSONL into train.bin and val.bin.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--tokenizer",
        choices=["byte", "byte-native", "bpe", "python-bpe", "native-bpe", "wordpiece", "sentencepiece"],
        default="byte",
    )
    p.add_argument("--tokenizer-path")
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--code-only", action="store_true")
    p.add_argument("--jsonl", action="store_true")
    p.add_argument("--jsonl-text-key", default="text")
    p.add_argument(
        "--mode",
        choices=["auto", "generative", "chat", "instruct", "completion", "code", "reasoning", "hybrid"],
        default="auto",
    )
    p.add_argument(
        "--loss-masking",
        choices=["auto", "none", "assistant-only", "assistant_only", "completion-only", "completion_only", "partial"],
        default="auto",
    )
    p.add_argument("--tokenizer-batch-size", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--min-chars", type=int, default=16)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--text-column", action="append", help="Structured text column(s) to use. Can be repeated.")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("inspect-dataset", help="Inspect formats, fields, and warnings for input data.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--text-key", default="text")
    p.add_argument("--limit", type=int, default=1000)
    p.set_defaults(func=cmd_inspect_dataset)

    p = sub.add_parser("validate-dataset", help="Validate readable text records before tokenization or packing.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--text-key", default="text")
    p.add_argument("--limit", type=int, default=1000)
    p.set_defaults(func=cmd_validate_dataset)

    p = sub.add_parser("clean-dataset", help="Clean, normalize, and deduplicate inputs into JSONL.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text-key", default="text")
    p.add_argument("--text-column", action="append")
    p.add_argument("--min-chars", type=int, default=16)
    p.add_argument("--max-chars", type=int)
    p.add_argument("--collapse-whitespace", action="store_true")
    p.add_argument("--no-deduplicate", action="store_true")
    p.add_argument("--near-deduplicate", action="store_true")
    p.add_argument("--language")
    p.set_defaults(func=cmd_clean_dataset)

    p = sub.add_parser("deduplicate-dataset", help="Deduplicate inputs into cleaned JSONL.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text-key", default="text")
    p.add_argument("--text-column", action="append")
    p.add_argument("--min-chars", type=int, default=16)
    p.add_argument("--max-chars", type=int)
    p.add_argument("--collapse-whitespace", action="store_true")
    p.add_argument("--near-deduplicate", action="store_true")
    p.add_argument("--language")
    p.set_defaults(no_deduplicate=False)
    p.set_defaults(func=cmd_clean_dataset)

    p = sub.add_parser("convert-dataset", help="Convert structured inputs to txt or JSONL text records.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["txt", "jsonl"], default="jsonl")
    p.add_argument("--text-key", default="text")
    p.add_argument("--text-column", action="append")
    p.set_defaults(func=cmd_convert_dataset)

    p = sub.add_parser("tokenizer-status", help="Show native tokenizer acceleration status.")
    p.set_defaults(func=cmd_tokenizer_status)

    p = sub.add_parser("tokenizer-report", help="Measure tokenizer compression and vocabulary usage.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument(
        "--tokenizer",
        choices=["byte", "byte-native", "bpe", "python-bpe", "native-bpe", "wordpiece", "sentencepiece"],
        default="byte",
    )
    p.add_argument("--tokenizer-path")
    p.add_argument("--text-key", default="text")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out")
    p.set_defaults(func=cmd_tokenizer_report)

    p = sub.add_parser("benchmark-tokenizer", help="Benchmark tokenizer throughput and memory use.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument(
        "--tokenizer",
        choices=["byte", "byte-native", "bpe", "python-bpe", "native-bpe", "wordpiece", "sentencepiece"],
        default="byte-native",
    )
    p.add_argument("--tokenizer-path")
    p.add_argument("--text-key", default="text")
    p.add_argument("--text-column", action="append")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--add-bos", action="store_true")
    p.add_argument("--add-eos", action="store_true")
    p.set_defaults(func=cmd_benchmark_tokenizer)

    p = sub.add_parser("train", help="Train from YAML config.")
    p.add_argument("--config", required=True)
    p.add_argument("--dashboard", action="store_true", help="Start the live web dashboard beside training.")
    p.add_argument("--dashboard-host", default="127.0.0.1")
    p.add_argument("--dashboard-port", type=int, default=7860)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("dashboard", help="Run a live browser dashboard for a training run.")
    p.add_argument("--run", default="runs/tiny")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_dashboard)

    for name, help_text, func in [
        ("generate", "Generate one completion.", cmd_generate),
        ("chat", "Interactive chat.", cmd_chat),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--checkpoint")
        p.add_argument("--model", help="Imported model name or external model path/HuggingFace id.")
        p.add_argument("--prompt", default="")
        p.add_argument("--device", default="auto")
        p.add_argument("--max-new-tokens", type=int, default=256)
        p.add_argument(
            "--mode",
            choices=["balanced", "chat", "creative", "coding", "deterministic", "low_memory", "high_quality"],
            default="balanced",
        )
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--top-k", type=int, default=50)
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--min-p", type=float)
        p.add_argument("--repetition-penalty", type=float, default=1.0)
        p.add_argument("--frequency-penalty", type=float, default=0.0)
        p.add_argument("--presence-penalty", type=float, default=0.0)
        p.add_argument("--no-repeat-ngram-size", type=int, default=0)
        p.add_argument("--deterministic", action="store_true")
        p.add_argument("--no-repetition-stop", action="store_true")
        p.add_argument("--repetition-window", type=int, default=64)
        p.add_argument("--repetition-threshold", type=float, default=0.85)
        p.add_argument("--stop-token", action="append")
        p.add_argument("--mirostat", action="store_true")
        p.add_argument("--beams", type=int, default=1)
        p.set_defaults(func=func)

    p = sub.add_parser("serve", help="Run the FastAPI inference server.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("export", help="Export ONNX or GGUF manifest.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["onnx", "gguf"], required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("benchmark", help="Benchmark forward pass throughput.")
    p.add_argument("--config", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("profile-config", help="Estimate params, FLOPs, and memory from config.")
    p.add_argument("--config", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--bytes-per-param", type=int, default=2)
    p.set_defaults(func=cmd_profile_config)

    p = sub.add_parser("evaluate", help="Evaluate checkpoint loss, perplexity, and token accuracy.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
