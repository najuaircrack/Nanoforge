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


def cmd_train_tokenizer(args: argparse.Namespace) -> None:
    from nanoforge.data.tokenizer import (
        train_bpe_tokenizer,
        train_sentencepiece_tokenizer,
        train_wordpiece_tokenizer,
    )

    files: list[Path] = []
    for root in _paths(args.input):
        files.extend([p for p in root.rglob("*") if p.is_file()] if root.is_dir() else [root])
    if args.type == "bpe":
        train_bpe_tokenizer(files, args.out, args.vocab_size, args.min_frequency)
    elif args.type == "wordpiece":
        train_wordpiece_tokenizer(files, args.out, args.vocab_size, args.min_frequency)
    else:
        model_type = "unigram" if args.type == "unigram" else "bpe"
        train_sentencepiece_tokenizer(files, args.out, args.vocab_size, model_type=model_type)


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
        progress_callback=update_bar,
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
    if stats.issues:
        print("issues:")
        for issue in stats.issues[:20]:
            print(f"- {issue.kind}: {issue.source}: {issue.message}")


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
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        mirostat=args.mirostat,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    from nanoforge.generation.engine import GenerationEngine

    engine = GenerationEngine.from_checkpoint(args.checkpoint, device=args.device)
    if args.beams > 1:
        print(engine.beam_search(args.prompt, args.max_new_tokens, args.beams))
    else:
        print(engine.complete(args.prompt, args.max_new_tokens, _sampling_from_args(args)))


def cmd_chat(args: argparse.Namespace) -> None:
    from nanoforge.generation.engine import GenerationEngine

    engine = GenerationEngine.from_checkpoint(args.checkpoint, device=args.device)
    print("Nanoforge chat. Press Ctrl+C or submit an empty prompt to exit.")
    try:
        while True:
            prompt = input("\nuser> ").strip()
            if not prompt:
                return
            print("assistant> ", end="", flush=True)
            for chunk in engine.stream(prompt, args.max_new_tokens, _sampling_from_args(args)):
                print(chunk, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print()


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

    p = sub.add_parser("train-tokenizer", help="Train a BPE, WordPiece, or SentencePiece tokenizer.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--type", choices=["bpe", "wordpiece", "sentencepiece", "unigram"], default="bpe")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.set_defaults(func=cmd_train_tokenizer)

    p = sub.add_parser("prepare", help="Pack raw text/code/JSONL into train.bin and val.bin.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", choices=["byte", "bpe", "wordpiece", "sentencepiece"], default="byte")
    p.add_argument("--tokenizer-path")
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--code-only", action="store_true")
    p.add_argument("--jsonl", action="store_true")
    p.add_argument("--jsonl-text-key", default="text")
    p.add_argument("--min-chars", type=int, default=16)
    p.add_argument("--no-progress", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("inspect-dataset", help="Inspect formats, fields, and warnings for input data.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--text-key", default="text")
    p.add_argument("--limit", type=int, default=1000)
    p.set_defaults(func=cmd_inspect_dataset)

    p = sub.add_parser("tokenizer-report", help="Measure tokenizer compression and vocabulary usage.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--tokenizer", choices=["byte", "bpe", "wordpiece", "sentencepiece"], default="byte")
    p.add_argument("--tokenizer-path")
    p.add_argument("--text-key", default="text")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out")
    p.set_defaults(func=cmd_tokenizer_report)

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
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--prompt", default="")
        p.add_argument("--device", default="auto")
        p.add_argument("--max-new-tokens", type=int, default=256)
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--top-k", type=int, default=50)
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--repetition-penalty", type=float, default=1.0)
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
