"""
Nanoforge TUI — full-scope terminal UI using Textual.

Install: pip install -e .[tui]
Launch:  nanoforge tui [--run runs/my-run]
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
    from textual.reactive import reactive
    from textual.screen import Screen
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        Log,
        ProgressBar,
        Select,
        Sparkline,
        Static,
        TabbedContent,
        TabPane,
        TextArea,
    )
    _TEXTUAL_OK = True
except ImportError:
    _TEXTUAL_OK = False


def _require_textual():
    if not _TEXTUAL_OK:
        raise RuntimeError("Install TUI extras: pip install -e .[tui]  (requires textual>=0.50)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_metrics(run_dir: Path, limit: int = 500) -> list[dict]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _scan_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)


def _fmt(v, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


# ── Screens ────────────────────────────────────────────────────────────────────

class TrainingScreen(Screen):
    """Live training monitor: loss sparkline, metrics table, checkpoint list."""

    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def __init__(self, run_dir: Path):
        super().__init__()
        self._run_dir = run_dir
        self._rows: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="train-left"):
                yield Label(f"Run: {self._run_dir.name}", id="run-label")
                yield Label("Train loss", classes="stat-label")
                yield Sparkline([], id="loss-spark", summary_function=min)
                yield Label("Val loss", classes="stat-label")
                yield Sparkline([], id="val-spark", summary_function=min)
                yield Label("Grad norm", classes="stat-label")
                yield Sparkline([], id="grad-spark", summary_function=max)
            with Vertical(id="train-right"):
                yield Label("Recent steps", classes="stat-label")
                tbl = DataTable(id="metrics-tbl")
                tbl.add_columns("Step", "Loss", "Val Loss", "LR", "Grad Norm", "Tok/s")
                yield tbl
                yield Label("Checkpoints", classes="stat-label")
                ck_tbl = DataTable(id="ck-tbl")
                ck_tbl.add_columns("File", "Size MB")
                yield ck_tbl
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(2.0, self._poll)
        self._poll()

    def _poll(self) -> None:
        self._rows = _read_metrics(self._run_dir)
        self.app.call_from_thread(self._update_ui)

    def _update_ui(self) -> None:
        rows = self._rows
        if not rows:
            return
        # Sparklines
        losses = [r["loss"] for r in rows if "loss" in r and r.get("loss") is not None]
        val_losses = [r["val_loss"] for r in rows if r.get("val_loss") is not None]
        grads = [r["grad_norm"] for r in rows if r.get("grad_norm") is not None]
        if losses:
            self.query_one("#loss-spark", Sparkline).data = losses[-120:]
        if val_losses:
            self.query_one("#val-spark", Sparkline).data = val_losses[-60:]
        if grads:
            self.query_one("#grad-spark", Sparkline).data = grads[-120:]
        # Metrics table
        tbl = self.query_one("#metrics-tbl", DataTable)
        tbl.clear()
        for r in rows[-30:][::-1]:
            tbl.add_row(
                str(r.get("step", "?")),
                _fmt(r.get("loss")),
                _fmt(r.get("val_loss")),
                _fmt(r.get("lr"), 6),
                _fmt(r.get("grad_norm")),
                _fmt(r.get("tokens_per_sec"), 0),
            )
        # Checkpoints
        ck_tbl = self.query_one("#ck-tbl", DataTable)
        ck_tbl.clear()
        for p in _scan_checkpoints(self._run_dir):
            ck_tbl.add_row(p.name, f"{p.stat().st_size / 1e6:.1f}")

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
                from nanoforge.data.profiler import DatasetProfiler
                prof = DatasetProfiler().profile(path)
                self.app.call_from_thread(log.write_line, str(vars(prof) if hasattr(prof, "__dict__") else prof))
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
        cpu_spark.data = list(cpu_spark.data or []) + [cpu]
        ram_spark.data = list(ram_spark.data or []) + [ram]
        self.query_one("#sys-text", Static).update(
            f"CPU: {cpu:.1f}%  RAM: {ram:.2f}/{vm.total/1e9:.2f} GB  "
            f"({vm.percent:.1f}% used)"
        )


# ── Main App ───────────────────────────────────────────────────────────────────

class NanoforgeApp(App):
    """Nanoforge TUI — full-scope terminal interface."""

    CSS = """
    Screen { background: $surface; }
    #train-left { width: 35%; padding: 1; }
    #train-right { width: 65%; padding: 1; }
    #inf-top { height: 3; }
    #inf-params { height: 3; }
    #inf-input { height: 8; }
    #inf-output { height: 1fr; }
    #cfg-left { width: 30%; padding: 1; }
    #cfg-right { width: 70%; padding: 1; }
    #cfg-editor { height: 1fr; }
    .stat-label { color: $text-muted; text-style: bold; margin-top: 1; }
    DataTable { height: 1fr; }
    Log { height: 1fr; border: solid $primary; }
    Sparkline { height: 5; }
    """

    BINDINGS = [
        Binding("1", "switch_screen('training')", "Training"),
        Binding("2", "switch_screen('inference')", "Inference"),
        Binding("3", "switch_screen('data')", "Data"),
        Binding("4", "switch_screen('config')", "Config"),
        Binding("5", "switch_screen('system')", "System"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, run_dir: str = "runs/nanoforge"):
        super().__init__()
        self._run_dir = Path(run_dir)

    def on_mount(self) -> None:
        self.install_screen(TrainingScreen(self._run_dir), name="training")
        self.install_screen(InferenceScreen(self._run_dir), name="inference")
        self.install_screen(DataScreen(), name="data")
        self.install_screen(ConfigScreen(), name="config")
        self.install_screen(SystemScreen(), name="system")
        self.push_screen("training")

    def action_switch_screen(self, name: str) -> None:
        self.switch_screen(name)


def launch_tui(run_dir: str = "runs/nanoforge") -> None:
    _require_textual()
    NanoforgeApp(run_dir=run_dir).run()
