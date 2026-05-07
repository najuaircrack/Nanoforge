from __future__ import annotations

from pathlib import Path

from nanoforge.progress import json_safe, read_jsonl_tail


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nanoforge Dashboard</title>
  <style>
    :root {
      --bg0: #f4f0e7;
      --bg1: #edf3ef;
      --ink: #151515;
      --muted: #6b6258;
      --line: #d9d0c3;
      --panel: rgba(255, 253, 247, 0.9);
      --panel2: rgba(247, 250, 247, 0.86);
      --teal: #0f766e;
      --blue: #315f8c;
      --amber: #a96416;
      --red: #a52828;
      --shadow: 0 18px 48px rgba(31, 28, 24, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Aptos Mono", "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      background:
        linear-gradient(115deg, rgba(15,118,110,0.08), transparent 34%),
        linear-gradient(245deg, rgba(169,100,22,0.08), transparent 32%),
        linear-gradient(135deg, var(--bg0), var(--bg1));
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: end;
      padding: 26px clamp(16px, 4vw, 44px) 18px;
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0; font-size: clamp(25px, 4vw, 44px); letter-spacing: 0; }
    .sub { margin-top: 8px; color: var(--muted); font-size: 13px; word-break: break-all; }
    .status {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px 14px;
      min-width: 220px;
      box-shadow: var(--shadow);
    }
    .status b { display: block; font-size: 13px; margin-bottom: 7px; }
    .status span { color: var(--muted); font-size: 12px; }
    main { padding: 20px clamp(16px, 4vw, 44px) 42px; }
    .progressWrap {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }
    .progressTop { display: flex; justify-content: space-between; gap: 14px; color: var(--muted); font-size: 12px; margin-bottom: 10px; }
    .bar { height: 12px; border-radius: 999px; background: #e1d8cb; overflow: hidden; }
    .bar > div { height: 100%; width: 0%; background: linear-gradient(90deg, var(--teal), var(--blue)); transition: width 250ms ease; }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat, .chart, .table {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .stat { padding: 14px; min-height: 92px; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .value { font-size: 24px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .hint { margin-top: 6px; color: var(--muted); font-size: 11px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(520px, 100%), 1fr));
      gap: 14px;
    }
    .chart { padding: 14px; min-height: 318px; }
    .chartHeader { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
    .legend { color: var(--muted); font-size: 11px; }
    canvas { width: 100%; height: 250px; display: block; }
    .table { margin-top: 14px; padding: 14px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    td, th { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 600; }
    .empty {
      display: none;
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 28px;
      background: var(--panel2);
      color: var(--muted);
      margin-bottom: 16px;
    }
    @media (max-width: 720px) {
      header { grid-template-columns: 1fr; }
      .status { min-width: 0; }
      .value { font-size: 20px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Nanoforge Training</h1>
      <div class="sub" id="run">waiting for metrics...</div>
    </div>
    <div class="status">
      <b id="state">Waiting</b>
      <span id="updated">No metric rows yet</span>
    </div>
  </header>
  <main>
    <div class="empty" id="empty">No metrics for this run yet. Start training or wait for the first log interval.</div>
    <section class="progressWrap">
      <div class="progressTop"><span id="stepText">step - / -</span><span id="eta">ETA -</span></div>
      <div class="bar"><div id="progress"></div></div>
    </section>
    <section class="stats">
      <div class="stat"><div class="label">Train Loss</div><div class="value" id="train_loss">-</div><div class="hint">current optimization loss</div></div>
      <div class="stat"><div class="label">Validation Loss</div><div class="value" id="val_loss">-</div><div class="hint">latest eval checkpoint</div></div>
      <div class="stat"><div class="label">Perplexity</div><div class="value" id="ppl">-</div><div class="hint">lower is better</div></div>
      <div class="stat"><div class="label">Tokens/sec</div><div class="value" id="tok">-</div><div class="hint">training throughput</div></div>
      <div class="stat"><div class="label">Grad Norm</div><div class="value" id="grad">-</div><div class="hint">before -> after clip</div></div>
      <div class="stat"><div class="label">Learning Rate</div><div class="value" id="lrVal">-</div><div class="hint">warmup/cosine schedule</div></div>
    </section>
    <section class="grid">
      <div class="chart"><div class="chartHeader"><div class="label">Loss</div><div class="legend">train teal | val amber</div></div><canvas id="loss"></canvas></div>
      <div class="chart"><div class="chartHeader"><div class="label">Gradient Norm</div><div class="legend">before red | after teal</div></div><canvas id="gn"></canvas></div>
      <div class="chart"><div class="chartHeader"><div class="label">Throughput</div><div class="legend">tokens/sec</div></div><canvas id="throughput"></canvas></div>
      <div class="chart"><div class="chartHeader"><div class="label">Learning Rate</div><div class="legend">scheduler</div></div><canvas id="lr"></canvas></div>
    </section>
    <section class="table">
      <div class="label">Latest Events</div>
      <table><thead><tr><th>Step</th><th>Event</th><th>Metrics</th></tr></thead><tbody id="events"></tbody></table>
    </section>
  </main>
  <script>
    const fmt = (v, digits = 3) => Number.isFinite(v) ? v.toFixed(digits) : "-";
    const byId = (id) => document.getElementById(id);
    const colors = { teal: "#0f766e", amber: "#a96416", blue: "#315f8c", red: "#a52828", line: "#d9d0c3", muted: "#6b6258" };
    function points(rows, key) {
      return rows.filter(r => Number.isFinite(r[key])).map(r => ({x: r.step, y: r[key]}));
    }
    function latestNumber(rows, key) {
      for (let i = rows.length - 1; i >= 0; i--) if (Number.isFinite(rows[i][key])) return rows[i][key];
      return undefined;
    }
    function draw(canvas, series) {
      const ctx = canvas.getContext("2d");
      const cssW = Math.max(canvas.clientWidth, 1);
      const cssH = Math.max(canvas.clientHeight, 1);
      canvas.width = cssW * devicePixelRatio;
      canvas.height = cssH * devicePixelRatio;
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      ctx.strokeStyle = colors.line;
      ctx.lineWidth = 1;
      ctx.strokeRect(38, 10, cssW - 48, cssH - 34);
      const all = series.flatMap(s => s.data);
      if (!all.length) {
        ctx.fillStyle = colors.muted;
        ctx.font = "12px Consolas, monospace";
        ctx.fillText("waiting for data", 46, 36);
        return;
      }
      const minX = Math.min(...all.map(p => p.x));
      const maxX = Math.max(...all.map(p => p.x));
      const minY = Math.min(...all.map(p => p.y));
      const maxY = Math.max(...all.map(p => p.y));
      const sx = x => 38 + ((x - minX) / Math.max(maxX - minX, 1)) * (cssW - 48);
      const sy = y => 10 + (1 - ((y - minY) / Math.max(maxY - minY, 1e-9))) * (cssH - 34);
      for (const s of series) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        s.data.forEach((p, i) => i ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y)));
        ctx.stroke();
      }
      ctx.fillStyle = colors.muted;
      ctx.font = "11px Consolas, monospace";
      ctx.fillText(fmt(maxY), 4, 18);
      ctx.fillText(fmt(minY), 4, cssH - 18);
    }
    function age(ts) {
      if (!Number.isFinite(ts)) return "No metric rows yet";
      const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
      return s < 60 ? `${s}s ago` : `${Math.floor(s / 60)}m ${s % 60}s ago`;
    }
    async function refresh() {
      const res = await fetch("/api/metrics");
      const payload = await res.json();
      const rows = payload.rows || [];
      const last = rows[rows.length - 1] || {};
      const maxSteps = latestNumber(rows, "run/max_steps");
      const step = Number.isFinite(last.step) ? last.step : 0;
      const pct = Number.isFinite(maxSteps) ? Math.min(100, Math.max(0, (step / maxSteps) * 100)) : 0;
      byId("empty").style.display = rows.length ? "none" : "block";
      byId("run").textContent = `${payload.run} | ${payload.path}`;
      byId("state").textContent = rows.length ? (last.event === "done" ? "Finished" : "Running") : "Waiting";
      byId("updated").textContent = `updated ${age(last.time)}`;
      byId("stepText").textContent = `step ${step.toLocaleString()} / ${Number.isFinite(maxSteps) ? maxSteps.toLocaleString() : "-"}`;
      byId("progress").style.width = `${pct}%`;
      const tok = latestNumber(rows, "train/tokens_per_sec");
      byId("eta").textContent = Number.isFinite(tok) && Number.isFinite(maxSteps) && tok > 0
        ? `progress ${pct.toFixed(1)}%`
        : "ETA -";
      byId("train_loss").textContent = fmt(latestNumber(rows, "train/loss"));
      byId("val_loss").textContent = fmt(latestNumber(rows, "val/loss"));
      byId("ppl").textContent = fmt(latestNumber(rows, "val/perplexity"), 2);
      byId("tok").textContent = Number.isFinite(tok) ? Math.round(tok).toLocaleString() : "-";
      const before = latestNumber(rows, "train/grad_norm_before_clip");
      const after = latestNumber(rows, "train/grad_norm_after_clip");
      byId("grad").textContent = Number.isFinite(before) || Number.isFinite(after) ? `${fmt(before, 1)} -> ${fmt(after, 1)}` : "-";
      byId("lrVal").textContent = Number.isFinite(latestNumber(rows, "train/lr")) ? latestNumber(rows, "train/lr").toExponential(2) : "-";
      draw(byId("loss"), [
        {color: colors.teal, data: points(rows, "train/loss")},
        {color: colors.amber, data: points(rows, "val/loss")}
      ]);
      draw(byId("gn"), [
        {color: colors.red, data: points(rows, "train/grad_norm_before_clip")},
        {color: colors.teal, data: points(rows, "train/grad_norm_after_clip")}
      ]);
      draw(byId("throughput"), [{color: colors.blue, data: points(rows, "train/tokens_per_sec")}]);
      draw(byId("lr"), [{color: colors.teal, data: points(rows, "train/lr")}]);
      byId("events").innerHTML = rows.slice(-18).reverse().map(r => {
        const metrics = Object.entries(r).filter(([k]) => !["time", "event", "step"].includes(k)).slice(0, 7)
          .map(([k, v]) => `${k}: ${Number.isFinite(v) ? fmt(v) : (v ?? "-")}`).join("<br>");
        return `<tr><td>${r.step ?? "-"}</td><td>${r.event || "train"}</td><td>${metrics}</td></tr>`;
      }).join("");
    }
    refresh().catch(console.error);
    setInterval(() => refresh().catch(console.error), 1500);
  </script>
</body>
</html>
"""


def create_dashboard_app(run_dir: str | Path):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except Exception as exc:
        raise RuntimeError("Install serve extras to use dashboard: pip install -e .[serve]") from exc

    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.jsonl"
    app = FastAPI(title="Nanoforge Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/metrics")
    def metrics(limit: int = 1000):
        return json_safe(
            {
                "run": run_path.name,
                "path": str(metrics_path),
                "rows": read_jsonl_tail(metrics_path, limit=limit),
            }
        )

    @app.get("/favicon.ico")
    def favicon():
        return HTMLResponse("", status_code=204)

    return app


def serve_dashboard(run_dir: str | Path, host: str = "127.0.0.1", port: int = 7860) -> None:
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install serve extras to use dashboard: pip install -e .[serve]") from exc
    uvicorn.run(create_dashboard_app(run_dir), host=host, port=port)
