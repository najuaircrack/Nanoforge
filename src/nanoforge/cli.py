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
    import json
    from dataclasses import asdict

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


def cmd_rag_index(args: argparse.Namespace) -> None:
    from nanoforge.rag.indexer import build_index

    engine = None
    if args.backend == "model":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when --backend model is used")
        from nanoforge.generation.engine import GenerationEngine
        engine = GenerationEngine.from_checkpoint(args.checkpoint, device=args.device)

    progress = None
    cb = None
    if not args.no_progress:
        from tqdm import tqdm
        progress = tqdm(desc="rag-index", unit="chunks", dynamic_ncols=True)

        def cb(n: int) -> None:
            progress.update(max(0, n - progress.n))

    store = build_index(
        input_paths=_paths(args.input),
        out_dir=Path(args.out),
        backend=args.backend,
        max_chunk_chars=args.max_chunk_chars,
        overlap_chars=args.overlap_chars,
        text_key=args.text_key,
        text_columns=args.text_column or [],
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        idf_fit_limit=args.idf_fit_limit,
        progress_callback=cb,
        engine=engine,
        st_model=args.st_model,
        tfidf_vocab=args.vocab_size,
    )
    if progress is not None:
        progress.close()
    print(f"indexed: {store.meta.num_docs} chunks")
    print(f"out:     {args.out}")
    print(f"backend: {store.meta.backend}")
    print(f"dim:     {store.meta.dim}")


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


def cmd_profile_dataset(args: argparse.Namespace) -> None:
    import json

    from nanoforge.data.profiler import profile_dataset

    profile = profile_dataset(_paths(args.input), text_key=args.text_key, limit=args.limit)
    if args.json:
        print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
        return
    print(profile.summary_line())
    print(f"recommended_text_columns: {list(profile.recommended_text_columns)}")
    print(f"recommended_loss_masking: {profile.recommended_loss_masking}")
    print(f"turn_field: {profile.turn_field}")
    print(f"role_style: {profile.role_style}")
    print(f"mode_votes: {profile.mode_votes}")
    print(f"fields: {profile.fields}")
    for note in profile.notes:
        print(f"note: {note}")


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
    import json
    from dataclasses import asdict

    from nanoforge.data.tokenizer import load_tokenizer
    from nanoforge.data.tokenizer_metrics import evaluate_tokenizer, save_tokenizer_report

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_path)
    report = evaluate_tokenizer(tokenizer, _paths(args.input), text_key=args.text_key, limit=args.limit)
    if args.out:
        save_tokenizer_report(report, args.out)
    print(json.dumps(asdict(report), indent=2))


def cmd_tokenizer_status(args: argparse.Namespace) -> None:
    import json
    from dataclasses import asdict

    from nanoforge.data.native_tokenizer import native_tokenizer_status

    _ = args
    print(json.dumps(asdict(native_tokenizer_status()), indent=2))


def cmd_benchmark_tokenizer(args: argparse.Namespace) -> None:
    import json
    from dataclasses import asdict

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


def cmd_pipeline(args: argparse.Namespace) -> None:
    from nanoforge.training.pipeline import load_recipe, run_pipeline
    recipe = load_recipe(args.config)
    if args.output_dir:
        recipe.output_dir = args.output_dir
    run_pipeline(recipe)


def cmd_pretrain(args: argparse.Namespace) -> None:
    """Convenience: single pretrain stage (plain CLM, no role masking)."""
    from nanoforge.config import load_config
    from nanoforge.training.trainer import Trainer
    cfg = load_config(args.config)
    cfg.training.mode = "pretrain"
    cfg.data.mode = "pretrain"
    cfg.data.loss_masking = "none"
    cfg.data.assistant_only_loss = False
    if args.steps:
        cfg.training.max_steps = args.steps
    if args.resume:
        cfg.training.resume_from_checkpoint = args.resume
    Trainer(cfg).train()


def cmd_dpo(args: argparse.Namespace) -> None:
    """Direct Preference Optimization fine-tuning."""
    from nanoforge.config import load_config
    from nanoforge.training.dpo_trainer import DPOTrainer

    cfg = load_config(args.config)
    DPOTrainer(cfg, args.data, beta=args.beta, reference_checkpoint=args.reference).train()


def cmd_distill(args: argparse.Namespace) -> None:
    """Knowledge distillation training."""
    from nanoforge.config import load_config
    from nanoforge.training.distillation_trainer import DistillationTrainer

    cfg = load_config(args.config)
    if args.teacher:
        cfg.training.teacher_checkpoint = args.teacher
    DistillationTrainer(cfg).train()


def cmd_train(args: argparse.Namespace) -> None:
    # A live terminal UI auto-launches when stdout is an interactive TTY (unless
    # disabled). When piped/redirected it stays headless with the plain tqdm bar.
    from nanoforge.training.trainer import run_training_with_optional_tui

    run_training_with_optional_tui(args.config, no_tui=getattr(args, "no_tui", False))


def cmd_web(args: argparse.Namespace) -> None:
    import webbrowser

    from nanoforge.webapp import serve_webapp

    url = f"http://{args.host}:{args.port}"
    print(f"web: {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    serve_webapp(args.run, args.host, args.port)


def cmd_tui(args: argparse.Namespace) -> None:
    from nanoforge.tui import launch_tui

    launch_tui(run_dir=args.run)


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
    if args.mode == "balanced":
        args.mode = "chat"
    engine = _load_generation_engine(args)
    system = args.system if args.system is not None else _default_system_prompt(engine)

    # Optional personal-facts layer: answer known factual questions from a small
    # retrieval KB before falling through to the (tiny) model. See rag/personal.py.
    answerer = None
    if getattr(args, "facts", None):
        from nanoforge.rag.personal import PersonalAnswerer

        answerer = PersonalAnswerer.load(
            args.facts, engine=engine, threshold=args.facts_threshold
        )
        if answerer is None:
            print(f"[facts] could not load index at {args.facts} — continuing without it")
        else:
            print(f"[facts] loaded personal KB from {args.facts}")

    print("Nanoforge chat. Press Ctrl+C or submit an empty prompt to exit.")
    if system:
        print(f"[system] {system}")
    try:
        while True:
            prompt = input("\nuser> ").strip()
            if not prompt:
                return
            print("assistant> ", end="", flush=True)
            # Personal question with a confident KB hit -> answer directly.
            fact = answerer.answer(prompt) if answerer is not None else None
            if fact is not None:
                print(fact)
                continue
            # This pipeline packs each assistant turn conditioned on the system
            # prompt + most recent user turn (see data/modes._chat_prefix), so we
            # condition inference the same way to match training.
            for chunk in engine.chat_stream(
                [("user", prompt)],
                args.max_new_tokens,
                _sampling_from_args(args),
                system=system,
            ):
                print(chunk, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print()


def _default_system_prompt(engine) -> str | None:
    cfg = getattr(engine, "config", None)
    if cfg is None or getattr(cfg.data, "mode", "") != "chat":
        return None
    # Chat models trained from an export with no system turn are hurt by an
    # out-of-distribution system prompt: feeding one measurably degrades
    # coherence. Config marks these with data.no_system_prompt.
    if getattr(cfg.data, "no_system_prompt", False):
        return None
    # Otherwise use the model's configured system prompt, if any. Must match what
    # the model was trained on, and stay short: the boundary packer truncates long
    # prefixes (see data/modes).
    prompt = getattr(cfg.data, "prompt_template", None)
    return prompt if isinstance(prompt, str) and prompt.strip() else None


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
        from nanoforge.data.profiler import profile_dataset

        profile = profile_dataset(input_paths, text_key=args.text_key, limit=args.inspect_limit)
        print(f"[0/4] inspecting dataset: {profile.summary_line()}")
        for note in profile.notes:
            print(f"       note: {note}")
        mode = profile.detected_mode
        if not text_columns and profile.recommended_text_columns:
            text_columns = profile.recommended_text_columns
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
    from nanoforge.export.onnx import export_onnx
    from nanoforge.generation.engine import GenerationEngine

    if args.format == "onnx":
        engine = GenerationEngine.from_checkpoint(args.checkpoint, device=args.device)
        export_onnx(engine.model, args.out)
    else:
        from nanoforge.export.gguf import export_gguf
        engine = GenerationEngine.from_checkpoint(args.checkpoint, device="cpu")
        export_gguf(engine.model, engine.config, args.out, dtype=getattr(args, "dtype", "f32"))


def cmd_benchmark(args: argparse.Namespace) -> None:
    from nanoforge.benchmark import benchmark_forward

    result = benchmark_forward(args.config, args.batch_size, args.steps, args.device)
    for key, value in result.items():
        print(f"{key}: {value:,.3f}" if isinstance(value, float) else f"{key}: {value}")


def cmd_profile_config(args: argparse.Namespace) -> None:
    import json
    from dataclasses import asdict

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


def cmd_evaluate_suite(args: argparse.Namespace) -> None:
    from nanoforge.evaluation.suite import (
        evaluate_task_suite,
        load_task_suite,
        write_evaluation_report,
    )

    suite = load_task_suite(args.suite)
    engine = _load_generation_engine(args)
    model_metadata: dict[str, object] = {"source": args.checkpoint or args.model}
    if args.checkpoint:
        from nanoforge.artifacts import load_artifact_manifest

        try:
            manifest = load_artifact_manifest(args.checkpoint)
            model_metadata.update(
                {
                    "artifact_id": manifest.artifact_id,
                    "content_fingerprint": manifest.content_fingerprint,
                    "artifact_kind": manifest.artifact_kind,
                }
            )
        except FileNotFoundError:
            model_metadata["manifest"] = None
    report = evaluate_task_suite(
        suite,
        engine,
        model_metadata=model_metadata,
        device=args.device,
        seed=args.seed,
    )
    paths = write_evaluation_report(report, args.out)
    print(f"suite: {report.suite_id}")
    print(f"passed: {report.summary['passed']}/{report.summary['total']}")
    print(f"failed: {report.summary['failed']}")
    print(f"errors: {report.summary['errors']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


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
            "head",
            "loss",
            "embedding",
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
    p.add_argument("--no-progress", action="store_true", help="Disable the merge progress bar.")
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

    p = sub.add_parser("rag-index", help="Build a RAG vector index from documents.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=["tfidf", "model", "sentence-transformers"], default="tfidf")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path (required for --backend model).")
    p.add_argument("--device", default="auto")
    p.add_argument("--st-model", default="all-MiniLM-L6-v2")
    p.add_argument("--max-chunk-chars", type=int, default=512)
    p.add_argument("--overlap-chars", type=int, default=64)
    p.add_argument("--text-key", default="text")
    p.add_argument("--text-column", action="append", help="Structured text column(s) to use. Can be repeated.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--idf-fit-limit", type=int, default=100_000)
    p.add_argument("--vocab-size", type=int, default=16_384)
    p.add_argument("--no-progress", action="store_true")
    p.set_defaults(func=cmd_rag_index)

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

    p = sub.add_parser("profile-dataset", help="Detect mode, text columns, and loss masking by inspecting structure.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--text-key", default="text")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--json", action="store_true", help="Emit the full profile as JSON.")
    p.set_defaults(func=cmd_profile_dataset)

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
    p.add_argument("--no-tui", action="store_true", help="Disable the live terminal UI even on an interactive TTY.")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("pipeline", help="Run a multi-stage training recipe (e.g. pretrain -> SFT -> DPO).")
    p.add_argument("--config", required=True, help="Recipe YAML (see configs/recipes/).")
    p.add_argument("--output-dir", help="Override the recipe's output_dir.")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("pretrain", help="Pretrain a base model on raw text (mode=pretrain, no loss masking).")
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, help="Override max_steps.")
    p.add_argument("--resume", help="Resume model weights from a checkpoint.")
    p.set_defaults(func=cmd_pretrain)

    p = sub.add_parser("dpo", help="Fine-tune with Direct Preference Optimization.")
    p.add_argument("--config", required=True, help="Training config YAML.")
    p.add_argument("--data", required=True, help="JSONL file with 'chosen' and 'rejected' fields.")
    p.add_argument("--beta", type=float, default=0.1, help="KL regularisation coefficient (default 0.1).")
    p.add_argument("--reference", help="Optional separate reference checkpoint (default: frozen copy of initial policy).")
    p.set_defaults(func=cmd_dpo)

    p = sub.add_parser("distill", help="Train a student model with knowledge distillation from a teacher.")
    p.add_argument("--config", required=True, help="Training config YAML (set teacher_checkpoint inside, or use --teacher).")
    p.add_argument("--teacher", help="Override config.training.teacher_checkpoint.")
    p.set_defaults(func=cmd_distill)

    p = sub.add_parser("tui", help="Launch the full-scope terminal UI.")
    p.add_argument("--run", default="runs/nanoforge", help="Run directory to monitor.")
    p.set_defaults(func=cmd_tui)

    p = sub.add_parser("web", help="Launch the full-scope web UI (React SPA + FastAPI).")
    p.add_argument("--run", default="runs/nanoforge", help="Run directory to monitor.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser.")
    p.set_defaults(func=cmd_web)

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
        p.add_argument("--system", default=None, help="System prompt for chat mode (overrides the default).")
        p.add_argument(
            "--facts",
            default=None,
            help="Path to a personal-facts RAG index (built from a question->answer "
            "facts file; see docs). When set, personal questions are answered from "
            "the facts KB before the model.",
        )
        p.add_argument(
            "--facts-threshold",
            type=float,
            default=0.5,
            help="Minimum retrieval similarity to answer from facts (else defer to the model).",
        )
        p.add_argument("--mirostat", action="store_true")
        p.add_argument("--beams", type=int, default=1)
        p.set_defaults(func=func)

    p = sub.add_parser("serve", help="Run the FastAPI inference server.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("export", help="Export model to GGUF (binary) or ONNX format.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["onnx", "gguf"], required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["f32", "f16"], default="f32", help="Tensor dtype for GGUF export.")
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

    p = sub.add_parser("evaluate-suite", help="Evaluate a model against a JSONL task suite.")
    p.add_argument("--suite", required=True, help="JSONL task suite.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Nanoforge checkpoint path.")
    source.add_argument("--model", help="Imported model name, path, or HuggingFace id.")
    p.add_argument("--out", required=True, help="Output directory or report.json path.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.set_defaults(func=cmd_evaluate_suite)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
