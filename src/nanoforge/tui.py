"""
Nanoforge TUI — full-scope terminal UI using Textual.

Two entry points:
  * ``launch_tui(run_dir)``          — full multi-screen app (``nanoforge tui``).
  * ``launch_training_tui(run_dir)`` — monitor-only live board, auto-opened by
    ``nanoforge train`` on an interactive TTY (see trainer.run_training_with_optional_tui).

Install: pip install -e .[tui]
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Grid, Horizontal, Vertical
    from textual.screen import ModalScreen, Screen
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        Log,
        Select,
        Sparkline,
        Static,
        TextArea,
        TabbedContent,
        TabPane,
    )
    _TEXTUAL_OK = True
except ImportError:
    _TEXTUAL_OK = False
    # Fallbacks so the module imports without the optional [tui] extra.
    # The launch functions raise a helpful error via _require_textual() before use.
    App = Screen = ModalScreen = object  # type: ignore
    ComposeResult = object  # type: ignore

    class Binding:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


def _require_textual():
    if not _TEXTUAL_OK:
        raise RuntimeError("Install TUI extras: pip install -e .[tui]  (requires textual>=0.50)")


# ── Metric ingestion ────────────────────────────────────────────────────────────

def _read_rows(run_dir: Path, limit: int = 4000) -> list[dict]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


class MetricStore:
    """Folds the namespaced JSONL metric stream into a live display state.

    The trainer writes one row per event (train/eval/health/skip/start/...), each
    carrying a *subset* of the namespaced keys (``train/loss``, ``val/loss``,
    ``health/logit_entropy`` …). We keep the last non-null value seen per key,
    plus per-step history for sparklines, and compute derived metrics (ETA,
    progress, mean throughput, loss delta, skipped-step count).
    """

    #: sparkline histories we maintain
    _HIST_KEYS = ("train/loss", "val/loss", "train/grad_norm", "train/tokens_per_sec", "health/logit_entropy")

    def __init__(self) -> None:
        self.latest: dict[str, Any] = {}
        self.hist: dict[str, list[float]] = {k: [] for k in self._HIST_KEYS}
        self.step: int = 0
        self.first_time: float | None = None
        self.last_time: float | None = None
        self.skipped: int = 0
        self._prev_loss: float | None = None
        self.loss_delta: float = float("nan")
        self.skip_reason: str = ""
        self.run_name: str = ""

    def ingest(self, rows: list[dict]) -> None:
        # Recompute from scratch each poll: cheap for a few thousand rows and keeps
        # sparkline history and counters exact even if the file was reset/rotated.
        self.latest.clear()
        for k in self.hist:
            self.hist[k] = []
        self.step = 0
        self.first_time = self.last_time = None
        self.skipped = 0
        self._prev_loss = None
        self.loss_delta = float("nan")
        self.skip_reason = ""

        for row in rows:
            t = row.get("time")
            if isinstance(t, (int, float)):
                self.first_time = t if self.first_time is None else self.first_time
                self.last_time = t
            event = row.get("event")
            if event == "skip":
                self.skipped += 1
                self.skip_reason = str(row.get("train/skip_reason", self.skip_reason))
            if isinstance(row.get("step"), (int, float)):
                self.step = max(self.step, int(row["step"]))
            for key, value in row.items():
                if key in ("time", "event"):
                    continue
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                self.latest[key] = value
            # Histories: only from rows that actually carry the key
            if event in ("train", None):
                for hk in self.hist:
                    v = row.get(hk)
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        self.hist[hk].append(float(v))
            # Loss delta between last two train-loss observations
            tl = row.get("train/loss")
            if isinstance(tl, (int, float)) and math.isfinite(float(tl)):
                if self._prev_loss is not None:
                    self.loss_delta = float(tl) - self._prev_loss
                self._prev_loss = float(tl)

    # ── derived ──
    def get(self, key: str, default: Any = None) -> Any:
        return self.latest.get(key, default)

    @property
    def elapsed(self) -> float:
        if self.first_time is None or self.last_time is None:
            return 0.0
        return max(0.0, self.last_time - self.first_time)

    @property
    def max_steps(self) -> int:
        return int(self.latest.get("run/max_steps", 0) or 0)

    @property
    def progress(self) -> float:
        ms = self.max_steps
        return (self.step / ms) if ms else 0.0

    @property
    def eta(self) -> float:
        if self.step <= 0 or not self.max_steps:
            return float("nan")
        rate = self.elapsed / self.step
        return rate * max(0, self.max_steps - self.step)

    @property
    def mean_tokens_per_sec(self) -> float:
        toks = self.latest.get("train/tokens_seen")
        if isinstance(toks, (int, float)) and self.elapsed > 0:
            return float(toks) / self.elapsed
        return float("nan")


# ── formatting helpers ──────────────────────────────────────────────────────────

def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.2e}"
        return f"{v:.{decimals}f}"
    return str(v)


def _fmt_int(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_compact(v: Any) -> str:
    """Human-compact large numbers: 1.2M, 3.4B, 12.3k."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        n = float(v)
    except (TypeError, ValueError):
        return str(v)
    for scale, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suf}"
    return f"{n:.0f}"


def _fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.1f}%"


def _fmt_dur(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or math.isinf(seconds))):
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _scan_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)


# ── The 30-metric live board ─────────────────────────────────────────────────────

# (label, key-or-callable, formatter, css-accent). Callables receive the MetricStore.
_BOARD: list[tuple[str, list[tuple[str, Any, Callable[[Any], str]]]]] = [
    ("PROGRESS", [
        ("Step", lambda s: s.step, _fmt_int),
        ("Max steps", lambda s: s.max_steps, _fmt_int),
        ("Progress", lambda s: s.progress, _fmt_pct),
        ("Elapsed", lambda s: s.elapsed, _fmt_dur),
        ("ETA", lambda s: s.eta, _fmt_dur),
        ("Tokens seen", "train/tokens_seen", _fmt_compact),
        ("Tokens total", "run/total_tokens", _fmt_compact),
    ]),
    ("LOSS", [
        ("Train loss", "train/loss", lambda v: _fmt(v, 4)),
        ("Val loss", "val/loss", lambda v: _fmt(v, 4)),
        ("Best val", "val/best_loss", lambda v: _fmt(v, 4)),
        ("Loss Δ", lambda s: s.loss_delta, lambda v: _fmt(v, 4)),
        ("Perplexity", "val/perplexity", lambda v: _fmt(v, 2)),
        ("Loss mean(32)", "health/loss_recent_mean", lambda v: _fmt(v, 4)),
    ]),
    ("OPTIMIZATION", [
        ("LR", "train/lr", lambda v: _fmt(v, 6)),
        ("Grad norm", "train/grad_norm", lambda v: _fmt(v, 3)),
        ("Grad pre-clip", "train/grad_norm_before_clip", lambda v: _fmt(v, 3)),
        ("Grad post-clip", "train/grad_norm_after_clip", lambda v: _fmt(v, 3)),
        ("Loss scale", "train/loss_scale", lambda v: _fmt(v, 1)),
        ("Skipped steps", lambda s: s.skipped, _fmt_int),
        ("Grad med(32)", "health/grad_norm_recent_median", lambda v: _fmt(v, 3)),
    ]),
    ("THROUGHPUT", [
        ("Tokens/sec", "train/tokens_per_sec", lambda v: _fmt(v, 0)),
        ("Step time", "train/step_time_sec", lambda v: _fmt(v, 3)),
        ("Mean tok/s", lambda s: s.mean_tokens_per_sec, lambda v: _fmt(v, 0)),
    ]),
    ("HEALTH", [
        ("Logit entropy", "health/logit_entropy", lambda v: _fmt(v, 3)),
        ("Opt nonfinite", "health/optimizer_nonfinite_tensors", _fmt_int),
        ("Opt state elems", "health/optimizer_state_elements", _fmt_compact),
        ("Parameters", "run/parameters", _fmt_compact),
    ]),
    ("MEMORY / MODEL", [
        ("CUDA alloc MB", "memory/cuda_allocated_mb", lambda v: _fmt(v, 1)),
        ("CUDA resv MB", "memory/cuda_reserved_mb", lambda v: _fmt(v, 1)),
        ("Train FLOP/tok", "profile/train_flops_per_token", _fmt_compact),
        ("KV cache MB", "profile/kv_cache_memory_mb", lambda v: _fmt(v, 1)),
    ]),
]


class MetricTile(Static):
    """A single label/value cell in the board."""

    def __init__(self, label: str, tile_id: str) -> None:
        super().__init__(id=tile_id, classes="tile")
        self._label = label
        self.update_value("—")

    def update_value(self, value: str) -> None:
        self.update(f"[dim]{self._label}[/dim]\n[b]{value}[/b]")


class LiveBoard(Container):
    """The full live metric board: tiles + sparklines + checkpoint list."""

    def __init__(self, run_dir: Path, store: MetricStore | None = None) -> None:
        super().__init__()
        self._run_dir = run_dir
        self._store = store or MetricStore()
        # stable id per (group, label)
        self._tile_ids: dict[tuple[str, str], str] = {}
        n = 0
        for group, items in _BOARD:
            for label, _key, _fmt_fn in items:
                self._tile_ids[(group, label)] = f"tile-{n}"
                n += 1

    def compose(self) -> ComposeResult:
        yield Label("", id="board-title")
        with Vertical(id="spark-row"):
            with Horizontal(classes="spark-line"):
                yield Label("Train loss", classes="spark-label")
                yield Sparkline([], id="sp-loss", summary_function=min)
                yield Label("Val loss", classes="spark-label")
                yield Sparkline([], id="sp-val", summary_function=min)
            with Horizontal(classes="spark-line"):
                yield Label("Grad norm", classes="spark-label")
                yield Sparkline([], id="sp-grad", summary_function=max)
                yield Label("Tok/s", classes="spark-label")
                yield Sparkline([], id="sp-tps", summary_function=max)
        for group, items in _BOARD:
            yield Label(group, classes="group-title")
            grid = Grid(classes="tile-grid")
            grid.styles.grid_size_columns = min(len(items), 7)
            with grid:
                for label, _key, _fmt_fn in items:
                    yield MetricTile(label, self._tile_ids[(group, label)])
        yield Label("Checkpoints", classes="group-title")
        ck = DataTable(id="ck-tbl")
        ck.add_columns("File", "Size MB", "Modified")
        yield ck

    def refresh_from(self, store: MetricStore) -> None:
        self._store = store
        title = f"[b]{store.run_name or self._run_dir.name}[/b]  ·  step [b]{store.step:,}[/b]/{store.max_steps:,}"
        if store.skip_reason:
            title += f"  ·  [yellow]last skip: {store.skip_reason}[/yellow]"
        try:
            self.query_one("#board-title", Label).update(title)
        except Exception:
            return
        for group, items in _BOARD:
            for label, key, fmt_fn in items:
                if callable(key):
                    raw = key(store)
                else:
                    raw = store.get(key)
                try:
                    tile = self.query_one(f"#{self._tile_ids[(group, label)]}", MetricTile)
                    tile.update_value(fmt_fn(raw))
                except Exception:
                    pass
        # Sparklines
        for sid, hk in (("#sp-loss", "train/loss"), ("#sp-val", "val/loss"),
                        ("#sp-grad", "train/grad_norm"), ("#sp-tps", "train/tokens_per_sec")):
            data = store.hist.get(hk, [])
            if data:
                try:
                    self.query_one(sid, Sparkline).data = data[-160:]
                except Exception:
                    pass
        # Checkpoints
        try:
            ck = self.query_one("#ck-tbl", DataTable)
            ck.clear()
            for p in reversed(_scan_checkpoints(self._run_dir)):
                st = p.stat()
                ck.add_row(p.name, f"{st.st_size / 1e6:.1f}",
                           time.strftime("%H:%M:%S", time.localtime(st.st_mtime)))
        except Exception:
            pass


# ── Screens ────────────────────────────────────────────────────────────────────

class TrainingScreen(Screen):
    """Live training monitor board (used inside the full multi-screen app)."""

    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def __init__(self, run_dir: Path):
        super().__init__()
        self._run_dir = run_dir
        self._store = MetricStore()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield LiveBoard(self._run_dir, self._store)
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(2.0, self._poll)
        self._poll()

    def _poll(self) -> None:
        def _do():
            rows = _read_rows(self._run_dir)
            self._store.ingest(rows)
            self.app.call_from_thread(self._update)
        threading.Thread(target=_do, daemon=True).start()

    def _update(self) -> None:
        try:
            self.query_one(LiveBoard).refresh_from(self._store)
        except Exception:
            pass

    def action_refresh(self) -> None:
        self._poll()


class InferenceScreen(Screen):
    """Load a checkpoint and run generation or classification."""

    def __init__(self, run_dir: Path):
        super().__init__()
        self._run_dir = run_dir
        self._engine = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            with Horizontal(id="inf-top"):
                yield Select([], id="ck-select", prompt="Select checkpoint…")
                yield Button("Load", id="load-btn", variant="primary")
                yield Label("", id="inf-status")
            with Horizontal(id="inf-params"):
                yield Label("temp:")
                yield Input("0.8", id="inf-temp", placeholder="temperature")
                yield Label("top_k:")
                yield Input("50", id="inf-topk", placeholder="top_k")
                yield Label("max tokens:")
                yield Input("256", id="inf-max", placeholder="max_new_tokens")
            yield Label("Prompt / Input:", classes="stat-label")
            yield TextArea("", id="inf-input")
            with Horizontal():
                yield Button("Run", id="run-btn", variant="success")
                yield Button("Clear", id="clear-btn")
            yield Label("Output:", classes="stat-label")
            yield Log(id="inf-output", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_checkpoints()

    def _refresh_checkpoints(self) -> None:
        ckpts = _scan_checkpoints(self._run_dir)
        sel = self.query_one("#ck-select", Select)
        sel.set_options([(p.name, str(p)) for p in ckpts])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load-btn":
            self._load()
        elif event.button.id == "run-btn":
            self._run()
        elif event.button.id == "clear-btn":
            self.query_one("#inf-output", Log).clear()

    def _load(self) -> None:
        sel = self.query_one("#ck-select", Select)
        if sel.value is Select.BLANK:
            self.query_one("#inf-status", Label).update("Select a checkpoint first")
            return
        self.query_one("#inf-status", Label).update("Loading…")
        def _do():
            try:
                from nanoforge.generation.engine import GenerationEngine
                self._engine = GenerationEngine.from_checkpoint(str(sel.value))
                task = getattr(getattr(self._engine.model, "config", None), "task_type", "causal_lm")
                self.app.call_from_thread(
                    self.query_one("#inf-status", Label).update,
                    f"✓ {Path(str(sel.value)).name} [{task}]"
                )
            except Exception as exc:
                self.app.call_from_thread(
                    self.query_one("#inf-status", Label).update,
                    f"Error: {exc}"
                )
        threading.Thread(target=_do, daemon=True).start()

    def _run(self) -> None:
        if self._engine is None:
            self.query_one("#inf-output", Log).write_line("Load a model first.")
            return
        prompt = self.query_one("#inf-input", TextArea).text
        try:
            temp = float(self.query_one("#inf-temp", Input).value)
            top_k = int(self.query_one("#inf-topk", Input).value)
            max_new = int(self.query_one("#inf-max", Input).value)
        except ValueError:
            self.query_one("#inf-output", Log).write_line("Invalid parameter values.")
            return
        log = self.query_one("#inf-output", Log)
        task = getattr(getattr(self._engine.model, "config", None), "task_type", "causal_lm")
        def _do():
            try:
                if task in {"sequence_classification", "token_classification",
                            "reward_modeling", "value_estimation"}:
                    result = self._engine.classify(prompt)
                    lines = [f"label {k}: {v*100:.2f}%" for k, v in sorted(result.items(), key=lambda x: -x[1])]
                    self.app.call_from_thread(log.write_line, "\n".join(lines))
                else:
                    from nanoforge.generation.sampling import SamplingConfig
                    sc = SamplingConfig(temperature=temp, top_k=top_k)
                    text = self._engine.complete(prompt, max_new_tokens=max_new, sampling=sc)
                    self.app.call_from_thread(log.write_line, text or "(empty)")
            except Exception as exc:
                self.app.call_from_thread(log.write_line, f"Error: {exc}")
        threading.Thread(target=_do, daemon=True).start()


class ToolApprovalModal(ModalScreen):
    """Review one concrete schema-validated side effect."""
    def __init__(self, call, spec, cancellation=None):
        super().__init__()
        self.call = call
        self.spec = spec
        self.cancellation = cancellation

    def compose(self):
        with Vertical(id="approval-dialog"):
            yield Label(f"Approve {self.spec.side_effect}: {self.call.name}?")
            yield TextArea(json.dumps(self.call.arguments, ensure_ascii=False, indent=2), read_only=True, id="approval-details")
            with Horizontal():
                yield Button("Approve this action", id="harness-approve", variant="warning")
                yield Button("Deny", id="harness-deny", variant="primary")
                yield Button("Cancel run", id="harness-cancel-approval", variant="error")

    def on_button_pressed(self, event):
        if event.button.id in {"harness-approve", "harness-deny"}:
            self.dismiss(event.button.id == "harness-approve")
        elif event.button.id == "harness-cancel-approval":
            if self.cancellation is not None:
                self.cancellation.cancel("cancelled during operator approval")
            self.dismiss(False)


class TuiApprovalProvider:
    def __init__(self, screen, cancellation, budget):
        self.screen, self.cancellation, self.budget = screen, cancellation, budget

    def approve(self, call, spec):
        finished = threading.Event()
        decision = [False]

        def respond(value):
            decision[0] = value is True
            finished.set()

        try:
            self.screen.app.call_from_thread(self.screen.app.push_screen, ToolApprovalModal(call, spec, self.cancellation), respond)
        except RuntimeError:
            return False
        while not finished.wait(0.05):
            if self.cancellation.cancelled or time.monotonic() >= self.budget.started_at + self.budget.max_wall_seconds:
                self.screen._post(self.screen._close_approval)
                return False
        return decision[0]


class HarnessScreen(Screen):
    """Local model/manual console with visible context, approvals and replay."""
    def __init__(self, project_path: str | None = None):
        super().__init__()
        self._project_path = project_path or "configs/project-example.yaml"
        self._trace = None
        self._harness_job_running = False
        self._last_result = None
        self._cancellation = None
        self._budget = None
        self._manifest = None
        self._workspace = None
        self._provider = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="harness-wrap"):
            with Horizontal(id="harness-project-row"):
                yield Input(self._project_path, id="harness-project", placeholder="Project manifest")
                yield Button("Load / Run", id="harness-run-btn", variant="primary")
                yield Button("Cancel", id="harness-cancel-btn", variant="error")
            yield Label("Request", classes="stat-label")
            yield TextArea("Inspect the workspace and report what you find.", id="harness-request")
            yield Label("Manual actions (used only when provider_kind is manual)", classes="stat-label")
            yield TextArea(
                '[{"schema_version":1,"type":"tool_call","name":"repo.list_files","arguments":{"pattern":"**/*","max_entries":50}},'
                '{"schema_version":1,"type":"final","content":"Manual inspection complete."}]',
                id="harness-actions", language="json",
            )
            yield Label("No run yet. Project controls model, context, tools and approval policy.", id="harness-status")
            yield Label("Budgets: no active run", id="harness-budgets")
            with Horizontal(id="harness-replay-row"):
                yield Input("", id="harness-trace-path", placeholder="Repository-relative saved trace")
                yield Button("Replay", id="harness-replay-btn")
                yield Button("Evaluate model", id="harness-eval-btn")
            with TabbedContent():
                with TabPane("Observations", id="harness-observations-tab"):
                    yield TextArea("", read_only=True, id="harness-observations")
                with TabPane("Context", id="harness-context-tab"):
                    yield TextArea("", read_only=True, id="harness-context")
                with TabPane("Trace", id="harness-trace-tab"):
                    yield Log(id="harness-log", highlight=True)
                with TabPane("Result / replay", id="harness-result-tab"):
                    yield TextArea("", read_only=True, id="harness-result")
                with TabPane("Evaluation", id="harness-eval-tab"):
                    yield TextArea("", read_only=True, id="harness-evaluation")
        yield Footer()

    def on_mount(self):
        self.set_interval(0.25, self._refresh_trace)

    def on_unmount(self):
        self._cancel()

    def _post(self, callback, *args):
        try:
            self.app.call_from_thread(callback, *args)
        except RuntimeError:
            pass

    def _close_approval(self):
        if isinstance(self.app.screen, ToolApprovalModal):
            self.app.pop_screen()

    def on_button_pressed(self, event):
        actions = {"harness-run-btn": self._run, "harness-cancel-btn": self._cancel,
                   "harness-replay-btn": self._replay, "harness-eval-btn": self._evaluate}
        callback = actions.get(event.button.id)
        if callback:
            callback()

    def _cancel(self):
        if self._cancellation is not None:
            self._cancellation.cancel("cancelled by operator")

    def _run(self):
        if self._harness_job_running:
            return
        project_path = self.query_one("#harness-project", Input).value.strip()
        request = self.query_one("#harness-request", TextArea).text.strip()
        raw_actions = self.query_one("#harness-actions", TextArea).text
        status = self.query_one("#harness-status", Label)
        if not project_path or not request:
            status.update("Project and request are required.")
            return
        from nanoforge.agent.protocol import CancellationToken
        self._cancellation = CancellationToken()
        self._harness_job_running = True
        self._trace = None
        self._last_result = None
        self.query_one("#harness-observations", TextArea).text = ""
        status.update("Loading explicitly selected local project…")

        def work():
            try:
                from nanoforge.agent.harness import AgentHarness
                from nanoforge.agent.project import build_project_provider, build_project_runtime, load_project_manifest, project_context_items, project_policy
                from nanoforge.agent.protocol import InMemoryTrace
                manifest = load_project_manifest(project_path)
                workspace, registry, budget = build_project_runtime(manifest, project_path)
                manual_actions = None
                if manifest.provider_kind == "manual":
                    manual_actions = json.loads(raw_actions)
                    if not isinstance(manual_actions, list) or not manual_actions:
                        raise ValueError("manual actions must be a non-empty JSON array")
                context = project_context_items(manifest, workspace)
                context_text = json.dumps(context, ensure_ascii=False)
                if len(context_text.encode("utf-8")) > manifest.budgets.get("max_context_bytes", 262144):
                    raise ValueError("selected context exceeds project context budget")
                self._manifest, self._workspace, self._budget = manifest, workspace, budget
                self._provider = build_project_provider(manifest, project_path, manual_actions)
                trace = InMemoryTrace()
                trace_relative = f"{manifest.trace_dir}/{trace.run_id}.json"
                trace_path = workspace.resolve(trace_relative)
                trace.path = trace_path
                trace.emit("project", "loaded", "manifest", {"project_fingerprint": manifest.fingerprint, "provider_kind": manifest.provider_kind})
                self._trace = trace
                self._post(self.query_one("#harness-trace-path", Input).__setattr__, "value", trace_relative)
                self._post(self.query_one("#harness-context", TextArea).__setattr__, "text", context_text)
                approval = TuiApprovalProvider(self, self._cancellation, budget)
                policy, approver = project_policy(manifest, approval)
                result = AgentHarness(registry).run(request, self._provider, budget=budget, context_items=context,
                    cancellation=self._cancellation, trace=trace,
                    executor_kwargs={"policy": policy, "approval": approver, "capabilities": set(manifest.capabilities)})
                self._last_result = result
                self._post(self._show_result, result)
            except Exception as exc:
                self._post(status.update, f"Harness error: {type(exc).__name__}: {exc}")
            finally:
                self._harness_job_running = False

        threading.Thread(target=work, daemon=True).start()

    def _show_result(self, result):
        self.query_one("#harness-status", Label).update(
            f"{result.status}: {self._manifest.name} · {self._manifest.provider_kind} · {len(result.trace)} trace events")
        self.query_one("#harness-observations", TextArea).text = json.dumps(result.observations, ensure_ascii=False, indent=2)
        self.query_one("#harness-result", TextArea).text = result.answer or result.error or result.status
        self._refresh_trace()

    def _refresh_trace(self):
        if self._budget is not None:
            budget = self._budget
            remaining = max(0.0, budget.max_wall_seconds - (time.monotonic() - budget.started_at))
            self.query_one("#harness-budgets", Label).update(
                f"Steps {budget.steps_used}/{budget.max_steps} · Tools {budget.tool_calls_used}/{budget.max_tool_calls} · "
                f"Tokens {getattr(budget, 'tokens_used', 0)}/{getattr(budget, 'max_tokens', '?')} · {remaining:.1f}s remaining")
        if self._trace is None:
            return
        log = self.query_one("#harness-log", Log)
        log.clear()
        for event in self._trace.to_list():
            details = json.dumps(event.get("details", {}), ensure_ascii=False, default=str)
            log.write_line(f"[{event['phase']}] {event['status']} {event['action']} {details}")

    def _replay(self):
        if self._harness_job_running:
            return
        try:
            from nanoforge.agent.harness import ReplayRunner
            from nanoforge.agent.protocol import InMemoryTrace
            from nanoforge.agent.project import build_project_runtime, load_project_manifest
            path = self.query_one("#harness-trace-path", Input).value.strip()
            if path:
                manifest_path = self.query_one("#harness-project", Input).value.strip()
                manifest = load_project_manifest(manifest_path)
                workspace, _, _ = build_project_runtime(manifest, manifest_path)
                trace = InMemoryTrace.load(workspace.resolve(path)).to_list()
            elif self._trace is not None:
                trace = self._trace.to_list()
            else:
                raise ValueError("select a saved trace or complete a run first")
            result = ReplayRunner().replay(trace)
            self.query_one("#harness-result", TextArea).text = json.dumps(result, ensure_ascii=False, indent=2)
            self.query_one("#harness-status", Label).update("Trace replayed without executing tools.")
        except Exception as exc:
            self.query_one("#harness-status", Label).update(f"Replay error: {exc}")

    def _evaluate(self):
        if self._harness_job_running:
            return
        if self._manifest is None or self._provider is None or self._manifest.provider_kind == "manual":
            self.query_one("#harness-status", Label).update("Run a project with a selected model before evaluation.")
            return
        if not self._manifest.eval_suite:
            self.query_one("#harness-status", Label).update("Configure eval_suite in the project manifest.")
            return
        from nanoforge.agent.protocol import CancellationToken
        self._cancellation = CancellationToken()
        self._harness_job_running = True
        self.query_one("#harness-status", Label).update("Evaluating the selected model against the configured suite…")
        def work():
            try:
                from nanoforge.agent.project import evaluate_project_model
                result = evaluate_project_model(self._manifest,
                    self.query_one("#harness-project", Input).value.strip(), cancellation=self._cancellation)
                report = result["report"]
                self._post(self.query_one("#harness-evaluation", TextArea).__setattr__, "text",
                           json.dumps(result, ensure_ascii=False, indent=2))
                self._post(self.query_one("#harness-status", Label).update,
                           f"Evaluation: {report['summary']['passed']}/{report['summary']['total']} passed; {report['summary']['errors']} errors")
            except Exception as exc:
                self._post(self.query_one("#harness-status", Label).update, f"Evaluation error: {exc}")
            finally:
                self._harness_job_running = False
        threading.Thread(target=work, daemon=True).start()



class DataScreen(Screen):
    """Browse data files and profile datasets."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            with Horizontal():
                yield Input("data", id="data-root", placeholder="data root dir")
                yield Button("List", id="list-btn", variant="primary")
                yield Button("Profile selected", id="prof-btn")
            yield DataTable(id="data-tbl")
            yield Log(id="data-log", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        tbl = self.query_one("#data-tbl", DataTable)
        tbl.add_columns("File", "Size KB")
        self._list()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "list-btn":
            self._list()
        elif event.button.id == "prof-btn":
            self._profile()

    def _list(self) -> None:
        root = Path(self.query_one("#data-root", Input).value or "data")
        tbl = self.query_one("#data-tbl", DataTable)
        tbl.clear()
        if not root.exists():
            return
        exts = {".txt", ".jsonl", ".json", ".csv", ".tsv", ".parquet", ".bin"}
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                try:
                    tbl.add_row(str(p), f"{p.stat().st_size / 1024:.1f}")
                except OSError:
                    pass

    def _profile(self) -> None:
        tbl = self.query_one("#data-tbl", DataTable)
        if tbl.cursor_row is None:
            return
        row = tbl.get_row_at(tbl.cursor_row)
        path = str(row[0])
        log = self.query_one("#data-log", Log)
        log.write_line(f"Profiling {path}…")
        def _do():
            try:
                from nanoforge.data.profiler import profile_dataset
                prof = profile_dataset([path])
                self.app.call_from_thread(log.write_line, json.dumps(prof.to_dict(), indent=2, default=str))
            except Exception as exc:
                self.app.call_from_thread(log.write_line, f"Error: {exc}")
        threading.Thread(target=_do, daemon=True).start()


class ConfigScreen(Screen):
    """View and edit YAML configs."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="cfg-left"):
                yield Label("Templates", classes="stat-label")
                yield DataTable(id="cfg-tbl")
                yield Button("Load selected", id="cfg-load-btn", variant="primary")
            with Vertical(id="cfg-right"):
                yield Label("YAML", classes="stat-label")
                yield TextArea("", id="cfg-editor", language="yaml")
                with Horizontal():
                    yield Input("configs/new.yaml", id="cfg-save-path")
                    yield Button("Save", id="cfg-save-btn", variant="success")
                    yield Label("", id="cfg-save-st")
        yield Footer()

    def on_mount(self) -> None:
        tbl = self.query_one("#cfg-tbl", DataTable)
        tbl.add_columns("Config")
        for p in sorted(Path("configs").glob("*.yaml")) if Path("configs").exists() else []:
            tbl.add_row(str(p))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cfg-load-btn":
            tbl = self.query_one("#cfg-tbl", DataTable)
            if tbl.cursor_row is not None:
                row = tbl.get_row_at(tbl.cursor_row)
                try:
                    text = Path(str(row[0])).read_text(encoding="utf-8")
                    self.query_one("#cfg-editor", TextArea).load_text(text)
                except Exception as exc:
                    self.query_one("#cfg-save-st", Label).update(str(exc))
        elif event.button.id == "cfg-save-btn":
            path = Path(self.query_one("#cfg-save-path", Input).value)
            text = self.query_one("#cfg-editor", TextArea).text
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                self.query_one("#cfg-save-st", Label).update(f"✓ saved {path}")
            except Exception as exc:
                self.query_one("#cfg-save-st", Label).update(f"Error: {exc}")


class SystemScreen(Screen):
    """CPU / RAM / GPU system stats."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Label("CPU %", classes="stat-label")
            yield Sparkline([], id="cpu-spark", summary_function=max)
            yield Label("RAM GB", classes="stat-label")
            yield Sparkline([], id="ram-spark", summary_function=max)
            yield Static("", id="sys-text")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(2.0, self._poll)
        self._poll()

    def _poll(self) -> None:
        def _do():
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0.2)
                vm = psutil.virtual_memory()
                ram = vm.used / 1e9
                self.app.call_from_thread(self._update, cpu, ram, vm)
            except ImportError:
                self.app.call_from_thread(
                    self.query_one("#sys-text", Static).update,
                    "psutil not installed — run: pip install psutil"
                )
        threading.Thread(target=_do, daemon=True).start()

    def _update(self, cpu: float, ram: float, vm) -> None:
        cpu_spark = self.query_one("#cpu-spark", Sparkline)
        ram_spark = self.query_one("#ram-spark", Sparkline)
        cpu_spark.data = (list(cpu_spark.data or []) + [cpu])[-160:]
        ram_spark.data = (list(ram_spark.data or []) + [ram])[-160:]
        self.query_one("#sys-text", Static).update(
            f"CPU: {cpu:.1f}%  RAM: {ram:.2f}/{vm.total/1e9:.2f} GB  "
            f"({vm.percent:.1f}% used)"
        )


class LaunchScreen(Screen):
    """Start a training run, pipeline, or auto-train from the TUI."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="launch-wrap"):
            yield Label("Config file", classes="stat-label")
            yield Input("configs/nano-debug.yaml", id="launch-config", placeholder="path/to/config.yaml")
            yield Label("Mode", classes="stat-label")
            with Horizontal():
                yield Button("train", id="btn-train", variant="primary")
                yield Button("pretrain", id="btn-pretrain")
                yield Button("pipeline", id="btn-pipeline")
                yield Button("auto-train", id="btn-auto")
            yield Label("", id="launch-status")
            yield Label("Output", classes="stat-label")
            yield Log(id="launch-log", highlight=True)
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mode_map = {
            "btn-train": "train",
            "btn-pretrain": "pretrain",
            "btn-pipeline": "pipeline",
            "btn-auto": "auto-train",
        }
        mode = mode_map.get(event.button.id)
        if mode:
            self._launch(mode)

    def _launch(self, mode: str) -> None:
        config = self.query_one("#launch-config", Input).value.strip()
        if not config:
            self.query_one("#launch-status", Label).update("Set a config path first.")
            return
        log = self.query_one("#launch-log", Log)
        status = self.query_one("#launch-status", Label)
        status.update(f"Starting {mode}…")

        def _do():
            import subprocess
            import sys
            # --no-tui: we already have a UI; don't let the child spawn another.
            cmd = [sys.executable, "-m", "nanoforge", mode, "--config", config]
            if mode == "train":
                cmd.append("--no-tui")
            self.app.call_from_thread(log.write_line, f"$ {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    self.app.call_from_thread(log.write_line, line.rstrip())
                proc.wait()
                self.app.call_from_thread(
                    status.update,
                    "✓ done" if proc.returncode == 0 else f"✗ exit {proc.returncode}"
                )
            except Exception as exc:
                self.app.call_from_thread(status.update, f"Error: {exc}")

        threading.Thread(target=_do, daemon=True).start()


# ── Shared CSS ───────────────────────────────────────────────────────────────────

_CSS = """
Screen { background: $surface; }
#train-left { width: 35%; padding: 1; }
#train-right { width: 65%; padding: 1; }
#inf-top { height: 3; }
#inf-params { height: 3; }
#inf-input { height: 8; }
#inf-output { height: 1fr; }
#harness-wrap { padding: 1; }
#harness-project-row { height: 3; }
#harness-project { width: 1fr; }
#harness-request { height: 5; }
#harness-actions { height: 8; }
#harness-log { height: 1fr; }
#harness-replay-row { height: 3; }
#harness-trace-path { width: 1fr; }
#harness-wrap TabbedContent { height: 1fr; }
#harness-wrap TabPane { padding: 0; }
#harness-wrap TabPane TextArea { height: 1fr; }
ToolApprovalModal { align: center middle; background: $background 70%; }
#approval-dialog { width: 90%; height: 80%; border: heavy $warning; padding: 1; }
#approval-details { height: 1fr; }
#cfg-left { width: 30%; padding: 1; }
#cfg-right { width: 70%; padding: 1; }
#cfg-editor { height: 1fr; }
#launch-wrap { padding: 1; }
#launch-log { height: 1fr; }
.stat-label { color: $text-muted; text-style: bold; margin-top: 1; }
DataTable { height: auto; max-height: 12; }
Log { height: 1fr; border: solid $primary; }
Sparkline { height: 3; width: 1fr; margin: 0 1; }
#board-title { padding: 0 1; height: 1; }
.group-title { color: $accent; text-style: bold; margin: 1 1 0 1; }
.spark-line { height: 3; }
.spark-label { width: 12; content-align: left middle; color: $text-muted; }
.tile-grid { height: auto; grid-gutter: 1; padding: 0 1; }
.tile {
    height: 3; padding: 0 1; content-align: left top;
    border: round $panel-lighten-2; background: $panel;
}
"""


# ── Full multi-screen app ────────────────────────────────────────────────────────

class NanoforgeApp(App):
    """Latticekin TUI — training, inference, and bounded harness console."""

    CSS = _CSS
    BINDINGS = [
        Binding("1", "switch_screen('launch')", "Launch"),
        Binding("2", "switch_screen('training')", "Monitor"),
        Binding("3", "switch_screen('inference')", "Inference"),
        Binding("4", "switch_screen('data')", "Data"),
        Binding("5", "switch_screen('config')", "Config"),
        Binding("6", "switch_screen('system')", "System"),
        Binding("7", "switch_screen('harness')", "Harness"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, run_dir: str = "runs/nanoforge", project: str | None = None):
        super().__init__()
        self._run_dir = Path(run_dir)
        self._project = project

    def on_mount(self) -> None:
        self.install_screen(LaunchScreen(), name="launch")
        self.install_screen(TrainingScreen(self._run_dir), name="training")
        self.install_screen(InferenceScreen(self._run_dir), name="inference")
        self.install_screen(DataScreen(), name="data")
        self.install_screen(ConfigScreen(), name="config")
        self.install_screen(SystemScreen(), name="system")
        self.install_screen(HarnessScreen(self._project), name="harness")
        self.push_screen("launch")

    def action_switch_screen(self, name: str) -> None:
        self.switch_screen(name)


# ── Monitor-only app (auto-launched beside `nanoforge train`) ─────────────────────

class TrainingMonitorApp(App):
    """Single-screen live board that tails ``metrics.jsonl`` for an active run.

    ``q`` detaches the UI (training keeps running headless); ``s`` requests a
    graceful stop of the attached trainer (checkpoints, then exits) if one was
    provided.
    """

    CSS = _CSS
    BINDINGS = [
        Binding("q", "quit", "Detach (keep training)"),
        Binding("s", "stop_training", "Stop training"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, run_dir: str, stop_callback: Callable[[], None] | None = None,
                 done_flag: Callable[[], bool] | None = None):
        super().__init__()
        self._run_dir = Path(run_dir)
        self._store = MetricStore()
        self._stop_callback = stop_callback
        self._done_flag = done_flag

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield LiveBoard(self._run_dir, self._store)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Latticekin · live training"
        self.set_interval(1.5, self._poll)
        self._poll()

    def _poll(self) -> None:
        def _do():
            rows = _read_rows(self._run_dir)
            self._store.ingest(rows)
            self.app.call_from_thread(self._update)
        threading.Thread(target=_do, daemon=True).start()

    def _update(self) -> None:
        try:
            self.query_one(LiveBoard).refresh_from(self._store)
        except Exception:
            pass
        # Auto-exit shortly after training reports completion.
        if self._done_flag is not None and self._done_flag():
            self.set_timer(2.0, self.exit)

    def action_stop_training(self) -> None:
        if self._stop_callback is not None:
            self._stop_callback()
            self.notify("Stop requested — trainer will checkpoint and exit.", severity="warning")

    def action_refresh(self) -> None:
        self._poll()


# ── Entry points ─────────────────────────────────────────────────────────────────

def launch_tui(run_dir: str = "runs/nanoforge", project: str | None = None) -> None:
    _require_textual()
    NanoforgeApp(run_dir=run_dir, project=project).run()


def launch_training_tui(
    run_dir: str,
    stop_callback: Callable[[], None] | None = None,
    done_flag: Callable[[], bool] | None = None,
) -> None:
    """Run the monitor-only live board. Blocks until the user detaches (q)."""
    _require_textual()
    TrainingMonitorApp(run_dir, stop_callback=stop_callback, done_flag=done_flag).run()
