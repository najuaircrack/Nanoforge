"""
Nanoforge Dashboard — complete rewrite
Drop-in replacement for the original dashboard.py

Fixes:
  · CPU/RAM charts now accumulate history server-side — no more empty charts on load
  · Chat checkpoint loader handles both old dict-config and new NanoforgeConfig checkpoints
  · Config loaded from run YAML + checkpoint .yaml sidecar
  · Sharp design: no rounded edges, no gradients, solid flat colors
  · All stat panels, multi-chart view, health alerts, events log, config viewer

Usage:
    nanoforge serve <run_dir>
    nanoforge serve <run_dir> --port 8080
"""

from __future__ import annotations

import collections
import time
from pathlib import Path

from nanoforge.progress import json_safe, read_jsonl_tail

# ── Server-side ring buffers for system history ────────────────────────────
_SYS_HISTORY_MAX = 180  # 6 min @ 2s interval
_sys_cpu_history: collections.deque[float] = collections.deque(maxlen=_SYS_HISTORY_MAX)
_sys_ram_history: collections.deque[float] = collections.deque(maxlen=_SYS_HISTORY_MAX)

# ──────────────────────────────────────────────────────────────────────────────
# HTML — single-file dashboard
# Sharp aesthetic: no border-radius, no gradients, solid flat fills
# ──────────────────────────────────────────────────────────────────────────────

HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Nanoforge · Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

/* ── Dark theme (default) ── */
:root{
  --bg:     #000000;
  --bg1:    #0a0a0a;
  --bg2:    #111111;
  --bg3:    #1a1a1a;
  --bg4:    #222222;
  --line:   rgba(255,255,255,0.08);
  --line2:  rgba(255,255,255,0.15);
  --ink:    #f0f0f0;
  --ink2:   #888888;
  --ink3:   #444444;
  --a:      #3b82f6;
  --a-hi:   #60a5fa;
  --teal:   #14b8a6;
  --amber:  #f59e0b;
  --red:    #ef4444;
  --green:  #22c55e;
  --purple: #a855f7;
  --pink:   #ec4899;
  --shadow: 0 4px 24px rgba(0,0,0,0.6);
  --shadow-sm: 0 2px 8px rgba(0,0,0,0.4);
  --r:      12px;
  --r-sm:   8px;
  --r-xs:   6px;
  --mono:'JetBrains Mono',monospace;
  --sans:'Inter',system-ui,sans-serif;
  --sw:220px;
  --hh:54px;
}

/* ── Light theme ── */
[data-theme="light"]{
  --bg:     #f8f9fc;
  --bg1:    #ffffff;
  --bg2:    #f1f3f8;
  --bg3:    #e8ebf2;
  --bg4:    #dde1ec;
  --line:   rgba(0,0,0,0.07);
  --line2:  rgba(0,0,0,0.13);
  --ink:    #0f0f0f;
  --ink2:   #6b7280;
  --ink3:   #9ca3af;
  --a:      #2563eb;
  --a-hi:   #3b82f6;
  --teal:   #0d9488;
  --amber:  #d97706;
  --red:    #dc2626;
  --green:  #16a34a;
  --purple: #7c3aed;
  --shadow: 0 4px 24px rgba(0,0,0,0.08);
  --shadow-sm: 0 2px 8px rgba(0,0,0,0.05);
}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:99px}

body{
  font-family:var(--sans);background:var(--bg);color:var(--ink);
  min-height:100vh;font-size:13px;line-height:1.5;
}

/* ── Sidebar ── */
#sb{
  position:fixed;left:0;top:0;bottom:0;width:var(--sw);
  background:var(--bg1);border-right:1px solid var(--line);
  display:flex;flex-direction:column;z-index:100;
  transition:transform .25s cubic-bezier(.4,0,.2,1);
}
.logo{
  padding:18px 16px 14px;border-bottom:1px solid var(--line);
  display:flex;align-items:center;gap:11px;
}
.logo-sq{
  width:30px;height:30px;background:var(--a);border-radius:var(--r-sm);
  display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:11px;font-weight:700;color:#fff;flex-shrink:0;
}
.logo-name{font-size:14px;font-weight:600;letter-spacing:-.3px}
.logo-ver{font-size:10px;color:var(--ink3);font-family:var(--mono);margin-top:1px}

.nav{padding:10px 8px;flex:1;overflow-y:auto}
.nav-sect{
  font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink3);padding:10px 10px 5px;font-weight:600;
}
.nav-item{
  display:flex;align-items:center;gap:9px;
  padding:8px 10px;border-radius:var(--r-sm);
  color:var(--ink2);cursor:pointer;font-size:12.5px;font-weight:500;
  transition:background .15s,color .15s;
}
.nav-item:hover{background:var(--bg2);color:var(--ink)}
.nav-item.active{background:var(--a);color:#fff}
[data-theme="light"] .nav-item.active{color:#fff}
.nav-item .ico{font-size:14px;width:16px;text-align:center;flex-shrink:0}
.nbadge{
  margin-left:auto;font-size:9px;padding:2px 7px;border-radius:99px;
  background:var(--bg3);color:var(--ink2);font-weight:600;font-family:var(--mono);
}
.nav-item.active .nbadge{background:rgba(255,255,255,0.2);color:#fff}

.sb-foot{
  padding:12px 14px;border-top:1px solid var(--line);
  font-size:11px;color:var(--ink3);font-family:var(--mono);
}
.sb-run{
  color:var(--ink2);font-weight:500;margin-bottom:4px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:11.5px;
}
.dot{
  display:inline-block;width:7px;height:7px;
  border-radius:50%;background:var(--ink3);margin-right:5px;vertical-align:middle;
  transition:background .4s;
}
.dot.run{background:var(--green);box-shadow:0 0 6px var(--green)}
.dot.done{background:var(--a)}

/* ── Main ── */
#main{margin-left:var(--sw);min-height:100vh;display:flex;flex-direction:column}

/* ── Topbar ── */
#tb{
  height:var(--hh);border-bottom:1px solid var(--line);
  padding:0 22px;display:flex;align-items:center;gap:12px;
  background:var(--bg1);position:sticky;top:0;z-index:90;
  backdrop-filter:blur(12px);
}
#tb h2{font-size:14px;font-weight:600;flex:1;letter-spacing:-.2px}
.tb-pill{
  font-family:var(--mono);font-size:10.5px;padding:4px 10px;border-radius:99px;
  background:var(--bg3);color:var(--ink2);border:1px solid var(--line);white-space:nowrap;
}
.btn-sq{
  width:32px;height:32px;border-radius:var(--r-sm);border:1px solid var(--line);
  background:transparent;color:var(--ink2);cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:14px;
  transition:background .15s,color .15s;
}
.btn-sq:hover{background:var(--bg2);color:var(--ink)}

/* ── Content ── */
#ct{padding:22px;flex:1}
.pg{display:none}.pg.on{display:block}

/* ── Progress ── */
.prog-wrap{
  background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);
  padding:16px 20px;margin-bottom:20px;box-shadow:var(--shadow-sm);
}
.prog-meta{
  display:flex;justify-content:space-between;
  font-family:var(--mono);font-size:11.5px;color:var(--ink2);margin-bottom:10px;
}
.prog-track{height:5px;background:var(--bg3);border-radius:99px;overflow:hidden}
.prog-fill{height:100%;width:0%;background:var(--a);border-radius:99px;transition:width .4s}
.prog-extra{
  display:flex;gap:20px;margin-top:10px;
  font-family:var(--mono);font-size:11px;color:var(--ink3);
}

/* ── Stat grid ── */
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:12px;margin-bottom:20px}
.sc{
  background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);
  padding:16px 18px;box-shadow:var(--shadow-sm);position:relative;overflow:hidden;
}
.sc::after{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:var(--ac,var(--a));border-radius:var(--r) var(--r) 0 0;
}
.sc-label{font-size:10px;color:var(--ink3);font-weight:600;letter-spacing:.07em;
  text-transform:uppercase;margin-bottom:8px}
.sc-val{font-family:var(--mono);font-size:22px;font-weight:500;
  color:var(--ink);line-height:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sc-sub{font-size:10px;color:var(--ink3);margin-top:5px;font-family:var(--mono)}

/* ── Section label ── */
.sl{
  font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink3);margin:22px 0 12px;display:flex;align-items:center;gap:9px;
}
.sl::after{content:'';flex:1;height:1px;background:var(--line)}

/* ── Chart grid ── */
.cg{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;margin-bottom:20px}
.cc{
  background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);
  padding:18px 18px 14px;box-shadow:var(--shadow-sm);
}
.cc-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.cc-title{font-size:10.5px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--ink3)}
.cc-legend{font-size:10px;color:var(--ink3);font-family:var(--mono)}
canvas.sp{width:100%;height:170px;display:block}

/* ── System bars ── */
.sb-track{height:4px;background:var(--bg3);border-radius:99px;margin-top:7px;overflow:hidden}
.sb-fill{height:100%;width:0%;border-radius:99px;transition:width .5s,background .4s}

/* ── Events table ── */
.et{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11.5px}
.et th{
  text-align:left;padding:9px 14px;border-bottom:1px solid var(--line);
  color:var(--ink3);font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
}
.et td{padding:8px 14px;border-bottom:1px solid var(--line);color:var(--ink2);vertical-align:top}
.et tr:last-child td{border-bottom:none}
.et tr:hover td{background:var(--bg2)}
.tag{
  display:inline-block;font-size:9.5px;padding:2px 8px;border-radius:99px;font-weight:600;
}
.t-tr{background:rgba(59,130,246,.12);color:var(--a)}
.t-ev{background:rgba(20,184,166,.12);color:var(--teal)}
.t-dn{background:rgba(34,197,94,.12);color:var(--green)}
.t-sp{background:rgba(168,85,247,.12);color:var(--purple)}
.t-he{background:rgba(245,158,11,.12);color:var(--amber)}
.t-wa{background:rgba(239,68,68,.12);color:var(--red)}

/* ── Checkpoint page ── */
.ck-grid{display:grid;gap:10px}
.ck-row{
  background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);
  padding:14px 18px;display:flex;align-items:center;gap:16px;cursor:pointer;
  transition:background .15s,border-color .15s,box-shadow .15s;
  box-shadow:var(--shadow-sm);
}
.ck-row:hover{background:var(--bg2);border-color:var(--a);box-shadow:0 0 0 1px var(--a)}
.ck-row.sel{border-color:var(--a);background:var(--bg2);box-shadow:0 0 0 1px var(--a)}
.ck-step{font-family:var(--mono);font-size:14px;font-weight:500;min-width:80px}
.ck-meta{flex:1;font-size:12px;color:var(--ink2)}
.ck-meta b{color:var(--ink);font-weight:600}
.ck-badge{
  font-size:9.5px;padding:3px 9px;border-radius:99px;font-weight:600;
  background:rgba(59,130,246,.12);color:var(--a);font-family:var(--mono);
}
.ck-detail{
  background:var(--bg2);border:1px solid var(--line);border-radius:var(--r);
  padding:18px;margin-top:14px;display:none;
  font-family:var(--mono);font-size:12px;line-height:2;color:var(--ink2);
}
.ck-detail.on{display:block}
.ck-dgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:4px}
.kv{display:flex;gap:8px}
.kk{color:var(--ink3);min-width:110px}
.kv2{color:var(--ink)}

/* ── Chat page ── */
#chat-msgs{
  min-height:300px;max-height:500px;overflow-y:auto;
  padding:12px 0;display:flex;flex-direction:column;gap:12px;margin-bottom:14px;
}
.msg{max-width:80%;padding:11px 15px;border-radius:var(--r);font-size:13px;line-height:1.6}
.msg-u{align-self:flex-end;background:var(--a);color:#fff}
.msg-m{
  align-self:flex-start;background:var(--bg2);color:var(--ink);
  border:1px solid var(--line);white-space:pre-wrap;font-family:var(--mono);font-size:12px;
}
.msg-s{align-self:center;font-size:11.5px;color:var(--ink3);font-style:italic;padding:4px 0;max-width:100%}
.ci-row{display:flex;gap:10px;align-items:flex-end}
#chat-in{
  flex:1;min-height:42px;max-height:120px;resize:none;
  background:var(--bg2);border:1px solid var(--line);border-radius:var(--r-sm);
  color:var(--ink);padding:10px 14px;font-family:var(--sans);font-size:13px;
  transition:border-color .15s,box-shadow .15s;outline:none;line-height:1.5;
}
#chat-in:focus{border-color:var(--a);box-shadow:0 0 0 3px rgba(59,130,246,.1)}
.btn{
  padding:10px 18px;border:none;border-radius:var(--r-sm);cursor:pointer;
  font-family:var(--sans);font-size:13px;font-weight:600;
  background:var(--a);color:#fff;transition:opacity .15s,transform .1s;
}
.btn:hover{opacity:.88}
.btn:active{transform:scale(.97)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-sec{background:var(--bg3);color:var(--ink);border:1px solid var(--line)}
.btn-sm{padding:6px 13px;font-size:12px}
.ck-sel{display:flex;align-items:center;gap:10px;margin-bottom:18px;flex-wrap:wrap}
.ck-sel select{
  flex:1;min-width:240px;background:var(--bg2);border:1px solid var(--line);border-radius:var(--r-sm);
  color:var(--ink);padding:9px 12px;font-family:var(--mono);font-size:12px;outline:none;
}
.ck-sel select:focus{border-color:var(--a)}
#chat-st{font-size:11px;color:var(--ink3);font-family:var(--mono);padding:3px 0}
.chat-cfg{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px;align-items:center;font-size:12.5px}
.chat-cfg label{display:flex;align-items:center;gap:6px;color:var(--ink2)}
.chat-cfg select,.chat-cfg input[type=number]{
  background:var(--bg2);border:1px solid var(--line);border-radius:var(--r-xs);
  color:var(--ink);padding:5px 9px;font-family:var(--mono);font-size:12px;outline:none;
}
.chat-cfg select:focus,.chat-cfg input[type=number]:focus{border-color:var(--a)}

/* ── Config page ── */
.cfg-sec{margin-bottom:24px}
.cfg-sec-title{
  font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink3);margin-bottom:10px;border-bottom:1px solid var(--line);padding-bottom:6px;
}
.cfg-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:4px}
.cfg-row{display:flex;justify-content:space-between;align-items:baseline;padding:7px 10px;border-radius:var(--r-xs)}
.cfg-row:hover{background:var(--bg2)}
.cfg-k{color:var(--ink2);font-size:12px;font-family:var(--mono)}
.cfg-v{color:var(--ink);font-size:12px;font-family:var(--mono);font-weight:500;text-align:right}

/* ── Alerts ── */
.al{padding:12px 15px;margin-bottom:10px;font-size:12.5px;
  display:flex;align-items:flex-start;gap:10px;border-radius:var(--r-sm);}
.al-w{background:rgba(245,158,11,.1);color:var(--amber);border:1px solid rgba(245,158,11,.2)}
.al-d{background:rgba(239,68,68,.1);color:var(--red);border:1px solid rgba(239,68,68,.2)}
.al-ok{background:rgba(34,197,94,.1);color:var(--green);border:1px solid rgba(34,197,94,.2)}
.al-ico{font-size:16px;flex-shrink:0;margin-top:1px}

/* ── Card ── */
.card{background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow-sm)}

/* ── Empty ── */
.empty{text-align:center;padding:56px 20px;color:var(--ink3);font-size:12.5px}
.empty-ico{font-size:38px;margin-bottom:12px}
.empty-txt{font-size:15px;font-weight:500;color:var(--ink2);margin-bottom:6px}

/* ── Responsive ── */
@media(max-width:860px){
  #sb{transform:translateX(-100%)}
  #sb.open{transform:translateX(0)}
  #main{margin-left:0}
  .cg{grid-template-columns:1fr}
  .sg{grid-template-columns:repeat(2,1fr)}
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.pls{animation:pulse 1.8s ease-in-out infinite}
</style>
</head>
<body>

<!-- Sidebar -->
<nav id="sb">
  <div class="logo">
    <div class="logo-sq">NF</div>
    <div>
      <div class="logo-name">Nanoforge</div>
      <div class="logo-ver">training lab</div>
    </div>
  </div>
  <div class="nav">
    <div class="nav-sect">Monitor</div>
    <div class="nav-item active" onclick="go('overview')"><span class="ico">⬡</span>Overview</div>
    <div class="nav-item" onclick="go('charts')"><span class="ico">∿</span>Charts</div>
    <div class="nav-item" onclick="go('system')"><span class="ico">◈</span>System</div>
    <div class="nav-item" onclick="go('health')"><span class="ico">♡</span>Health</div>
    <div class="nav-sect">Model</div>
    <div class="nav-item" onclick="go('checkpoints')">
      <span class="ico">◧</span>Checkpoints<span class="nbadge" id="ck-n">0</span>
    </div>
    <div class="nav-item" onclick="go('chat')"><span class="ico">◻</span>Chat</div>
    <div class="nav-sect">Run</div>
    <div class="nav-item" onclick="go('config')"><span class="ico">⊞</span>Config</div>
    <div class="nav-item" onclick="go('events')"><span class="ico">≡</span>Events</div>
  </div>
  <div class="sb-foot">
    <div class="sb-run" id="sb-run">—</div>
    <div><span class="dot" id="sb-dot"></span><span id="sb-st">Waiting</span></div>
    <div style="margin-top:4px" id="sb-age">—</div>
  </div>
</nav>

<!-- Main -->
<div id="main">
  <header id="tb">
    <button class="btn-sq" onclick="toggleSB()" aria-label="Menu">☰</button>
    <h2 id="tb-title">Overview</h2>
    <span class="tb-pill" id="tb-step">step — / —</span>
    <span class="tb-pill pls" id="tb-eta" style="display:none">ETA…</span>
    <button class="btn-sq" onclick="toggleTheme()" title="Toggle light/dark" aria-label="Theme">◑</button>
  </header>

  <div id="ct">

    <!-- ═══ OVERVIEW ═══ -->
    <div class="pg on" id="pg-overview">
      <div id="alert-zone"></div>
      <div class="prog-wrap">
        <div class="prog-meta">
          <span id="p-step">step — / —</span><span id="p-pct">0%</span>
        </div>
        <div class="prog-track"><div class="prog-fill" id="p-bar"></div></div>
        <div class="prog-extra">
          <span id="p-epoch">epoch —</span>
          <span id="p-elapsed">elapsed —</span>
          <span id="p-tok">— tok/s</span>
        </div>
      </div>

      <div class="sl">Training Metrics</div>
      <div class="sg">
        <div class="sc" style="--ac:var(--a)">
          <div class="sc-label">Train Loss</div>
          <div class="sc-val" id="s-tl">—</div>
          <div class="sc-sub">optimisation objective</div>
        </div>
        <div class="sc" style="--ac:var(--amber)">
          <div class="sc-label">Val Loss</div>
          <div class="sc-val" id="s-vl">—</div>
          <div class="sc-sub">latest eval</div>
        </div>
        <div class="sc" style="--ac:var(--teal)">
          <div class="sc-label">Perplexity</div>
          <div class="sc-val" id="s-ppl">—</div>
          <div class="sc-sub">e ^ val_loss</div>
        </div>
        <div class="sc" style="--ac:var(--purple)">
          <div class="sc-label">Tokens / sec</div>
          <div class="sc-val" id="s-tok">—</div>
          <div class="sc-sub">throughput</div>
        </div>
        <div class="sc" style="--ac:var(--red)">
          <div class="sc-label">Grad Norm</div>
          <div class="sc-val" id="s-gn">—</div>
          <div class="sc-sub">before → after clip</div>
        </div>
        <div class="sc" style="--ac:var(--green)">
          <div class="sc-label">Learning Rate</div>
          <div class="sc-val" id="s-lr">—</div>
          <div class="sc-sub">warmup / cosine</div>
        </div>
      </div>

      <div class="sl">Quick Charts</div>
      <div class="cg">
        <div class="cc">
          <div class="cc-head"><span class="cc-title">Loss</span><span class="cc-legend">blue=train · amber=val</span></div>
          <canvas class="sp" id="ov-loss"></canvas>
        </div>
        <div class="cc">
          <div class="cc-head"><span class="cc-title">Throughput</span><span class="cc-legend">tokens / sec</span></div>
          <canvas class="sp" id="ov-tok"></canvas>
        </div>
      </div>

      <div class="sl">Recent Events</div>
      <div class="card" style="padding:0;overflow:auto">
        <table class="et">
          <thead><tr><th>Step</th><th>Event</th><th>Loss</th><th>LR</th><th>Details</th></tr></thead>
          <tbody id="ov-ev"></tbody>
        </table>
      </div>
    </div>

    <!-- ═══ CHARTS ═══ -->
    <div class="pg" id="pg-charts">
      <div class="cg">
        <div class="cc"><div class="cc-head"><span class="cc-title">Loss Curves</span><span class="cc-legend">blue=train · amber=val</span></div><canvas class="sp" id="ch-loss"></canvas></div>
        <div class="cc"><div class="cc-head"><span class="cc-title">Gradient Norm</span><span class="cc-legend">red=before · teal=after clip</span></div><canvas class="sp" id="ch-gn"></canvas></div>
        <div class="cc"><div class="cc-head"><span class="cc-title">Throughput</span><span class="cc-legend">tokens / sec</span></div><canvas class="sp" id="ch-tput"></canvas></div>
        <div class="cc"><div class="cc-head"><span class="cc-title">Learning Rate</span><span class="cc-legend">scheduler curve</span></div><canvas class="sp" id="ch-lr"></canvas></div>
        <div class="cc"><div class="cc-head"><span class="cc-title">Val Perplexity</span><span class="cc-legend">lower is better</span></div><canvas class="sp" id="ch-ppl"></canvas></div>
        <div class="cc"><div class="cc-head"><span class="cc-title">CPU Usage</span><span class="cc-legend">% over time (server history)</span></div><canvas class="sp" id="ch-cpu"></canvas></div>
        <div class="cc"><div class="cc-head"><span class="cc-title">RAM Usage</span><span class="cc-legend">GB over time (server history)</span></div><canvas class="sp" id="ch-ram"></canvas></div>
      </div>
    </div>

    <!-- ═══ SYSTEM ═══ -->
    <div class="pg" id="pg-system">
      <div class="sg">
        <div class="sc" style="--ac:var(--a)">
          <div class="sc-label">CPU Usage</div>
          <div class="sc-val" id="sy-cpu">—</div>
          <div class="sb-track"><div class="sb-fill" id="sb-cpu"></div></div>
          <div class="sc-sub">% all cores</div>
        </div>
        <div class="sc" style="--ac:var(--teal)">
          <div class="sc-label">RAM</div>
          <div class="sc-val" id="sy-ram">—</div>
          <div class="sb-track"><div class="sb-fill" id="sb-ram"></div></div>
          <div class="sc-sub" id="sy-ram-total">used / total GB</div>
        </div>
        <div class="sc" style="--ac:var(--red)">
          <div class="sc-label">CPU Temp</div>
          <div class="sc-val" id="sy-temp">—</div>
          <div class="sb-track"><div class="sb-fill" id="sb-temp"></div></div>
          <div class="sc-sub">degrees celsius</div>
        </div>
        <div class="sc" style="--ac:var(--purple)">
          <div class="sc-label">CPU Freq</div>
          <div class="sc-val" id="sy-freq">—</div>
          <div class="sc-sub">GHz current</div>
        </div>
        <div class="sc" style="--ac:var(--amber)">
          <div class="sc-label">CPU Cores</div>
          <div class="sc-val" id="sy-cores">—</div>
          <div class="sc-sub">logical / physical</div>
        </div>
        <div class="sc" style="--ac:var(--green)">
          <div class="sc-label">Swap</div>
          <div class="sc-val" id="sy-swap">—</div>
          <div class="sb-track"><div class="sb-fill" id="sb-swap"></div></div>
          <div class="sc-sub">used / total GB</div>
        </div>
      </div>
      <div class="cg">
        <div class="cc"><div class="cc-head"><span class="cc-title">CPU %</span><span class="cc-legend">server-accumulated history</span></div><canvas class="sp" id="sy-cpu-ch"></canvas></div>
        <div class="cc"><div class="cc-head"><span class="cc-title">RAM GB</span><span class="cc-legend">server-accumulated history</span></div><canvas class="sp" id="sy-ram-ch"></canvas></div>
      </div>
    </div>

    <!-- ═══ HEALTH ═══ -->
    <div class="pg" id="pg-health">
      <div id="health-alerts">
        <div class="empty">
          <div class="empty-ico">♡</div>
          <div class="empty-txt">All looks healthy</div>
          <div>Start training to see diagnostics</div>
        </div>
      </div>
      <div class="sl">Gradient History</div>
      <div class="cc">
        <div class="cc-head"><span class="cc-title">Grad Norm Before Clip</span><span class="cc-legend">red zone = explosion risk</span></div>
        <canvas class="sp" id="h-gn"></canvas>
      </div>
    </div>

    <!-- ═══ CHECKPOINTS ═══ -->
    <div class="pg" id="pg-checkpoints">
      <div id="ck-list">
        <div class="empty">
          <div class="empty-ico">◧</div>
          <div class="empty-txt">No checkpoints yet</div>
          <div>Saved to &lt;run_dir&gt;/ckpt-*.pt once training saves them</div>
        </div>
      </div>
      <div class="ck-detail" id="ck-panel"></div>
    </div>

    <!-- ═══ CHAT ═══ -->
    <div class="pg" id="pg-chat">
      <div class="card" style="margin-bottom:16px;padding:20px">
        <div class="sc-label" style="margin-bottom:16px;font-size:12px">Model Chat</div>
        <div class="ck-sel">
          <select id="chat-ck"><option value="">— select checkpoint —</option></select>
          <button class="btn btn-sm" onclick="loadCkpt()">Load</button>
          <span id="chat-st">No checkpoint loaded</span>
        </div>
        <div class="chat-cfg">
          <label>Temperature<input type="number" id="cfg-t" value="0.8" step="0.05" min="0" max="2" style="width:62px"/></label>
          <label>Top-k<input type="number" id="cfg-k" value="50" step="1" min="1" max="500" style="width:62px"/></label>
          <label>Max tokens<input type="number" id="cfg-m" value="256" step="8" min="8" max="2048" style="width:70px"/></label>
          <label>Mode<select id="cfg-md">
            <option value="balanced">balanced</option>
            <option value="chat">chat</option>
            <option value="creative">creative</option>
            <option value="coding">coding</option>
            <option value="deterministic">deterministic</option>
            <option value="low_memory">low_memory</option>
            <option value="high_quality">high_quality</option>
          </select></label>
          <button class="btn btn-sec btn-sm" onclick="clearChat()">Clear</button>
        </div>
        <div id="chat-msgs">
          <div class="msg msg-s">Load a checkpoint to start chatting with your model.</div>
        </div>
        <div class="ci-row">
          <textarea id="chat-in" rows="1" placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            onkeydown="chatKey(event)"></textarea>
          <button class="btn" onclick="sendChat()" id="chat-btn">Send</button>
        </div>
      </div>
    </div>

    <!-- ═══ CONFIG ═══ -->
    <div class="pg" id="pg-config">
      <div id="cfg-body">
        <div class="empty">
          <div class="empty-ico">⊞</div>
          <div class="empty-txt">Config not loaded</div>
          <div>Read from run YAML once available</div>
        </div>
      </div>
    </div>

    <!-- ═══ EVENTS ═══ -->
    <div class="pg" id="pg-events">
      <div class="card" style="padding:0;overflow:auto">
        <table class="et">
          <thead><tr><th>Step</th><th>Time</th><th>Event</th><th>Loss</th><th>Val Loss</th><th>LR</th><th>Grad Norm</th></tr></thead>
          <tbody id="ev-body"></tbody>
        </table>
      </div>
    </div>

  </div>
</div>

<script>
// ── Theme ───────────────────────────────────────────────────────────────────
function toggleTheme(){
  const html=document.documentElement;
  html.dataset.theme = html.dataset.theme==='dark'?'light':'dark';
}

// ── Nav ─────────────────────────────────────────────────────────────────────
const TITLES={overview:'Overview',charts:'Charts',system:'System',
  health:'Health & Alerts',checkpoints:'Checkpoints',chat:'Chat',config:'Config',events:'Event Log'};
function go(id){
  document.querySelectorAll('.pg').forEach(p=>p.classList.remove('on'));
  document.getElementById('pg-'+id).classList.add('on');
  document.querySelectorAll('.nav-item').forEach(n=>{
    const txt=n.textContent.trim().toLowerCase();
    n.classList.toggle('active',txt.startsWith(id.replace('-',' ').split(' ')[0]));
  });
  document.getElementById('tb-title').textContent=TITLES[id]||id;
  if(id==='charts') drawAll(R);
  if(id==='health') drawAll(R);
  if(id==='system') drawSysFull();
  if(id==='checkpoints') renderCkpts();
  if(id==='config') renderCfg();
}
function toggleSB(){document.getElementById('sb').classList.toggle('open')}

// ── Chart helpers ────────────────────────────────────────────────────────────
function n(v){return typeof v==='number'&&isFinite(v)}
function last(rows,key){for(let i=rows.length-1;i>=0;i--)if(n(rows[i][key]))return rows[i][key];}
function fmt(v,d=3){return n(v)?Number(v).toFixed(d):'-';}
function pts(rows,key){return rows.filter(r=>n(r[key])).map(r=>({x:r.step,y:r[key]}))}

// Theme-aware chart colors
function isDark(){return document.documentElement.dataset.theme!=='light'}
function gridCol(){return isDark()?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)'}
function textCol(){return isDark()?'rgba(255,255,255,0.3)':'rgba(0,0,0,0.35)'}

const C={a:'#3b82f6',teal:'#14b8a6',amber:'#f59e0b',red:'#ef4444',green:'#22c55e',purple:'#a855f7'};

function draw(cid,series){
  const cv=document.getElementById(cid); if(!cv) return;
  const ctx=cv.getContext('2d');
  const W=cv.clientWidth||400, H=cv.clientHeight||170;
  cv.width=W*devicePixelRatio; cv.height=H*devicePixelRatio;
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  ctx.clearRect(0,0,W,H);
  const P={l:44,r:12,t:10,b:28};
  const cW=W-P.l-P.r, cH=H-P.t-P.b;
  ctx.strokeStyle=gridCol(); ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){const y=P.t+cH*(i/4);ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(P.l+cW,y);ctx.stroke()}
  const all=series.flatMap(s=>s.data);
  if(!all.length){
    ctx.fillStyle=textCol();ctx.font='11px JetBrains Mono,monospace';
    ctx.fillText('waiting for data…',P.l+8,P.t+cH/2+4);return;
  }
  const minX=Math.min(...all.map(p=>p.x)), maxX=Math.max(...all.map(p=>p.x));
  const minY=Math.min(...all.map(p=>p.y)), maxY=Math.max(...all.map(p=>p.y));
  const sx=x=>P.l+((x-minX)/Math.max(maxX-minX,1))*cW;
  const sy=y=>P.t+(1-(y-minY)/Math.max(maxY-minY,1e-9))*cH;
  for(const s of series){
    ctx.strokeStyle=s.color;ctx.lineWidth=1.8;ctx.lineJoin='round';ctx.beginPath();
    s.data.forEach((p,i)=>i?ctx.lineTo(sx(p.x),sy(p.y)):ctx.moveTo(sx(p.x),sy(p.y)));
    ctx.stroke();
    // last point dot
    if(s.data.length){
      const lp=s.data[s.data.length-1];
      ctx.fillStyle=s.color;ctx.beginPath();ctx.arc(sx(lp.x),sy(lp.y),3,0,Math.PI*2);ctx.fill();
    }
  }
  const tc=textCol();
  ctx.fillStyle=tc;ctx.font='9.5px JetBrains Mono,monospace';
  ctx.textAlign='right';
  ctx.fillText(maxY.toFixed(3),P.l-4,P.t+7);
  ctx.fillText(minY.toFixed(3),P.l-4,P.t+cH+2);
  ctx.textAlign='left';ctx.fillText(minX.toLocaleString(),P.l,H-5);
  ctx.textAlign='right';ctx.fillText(maxX.toLocaleString(),P.l+cW,H-5);
}

// ── State ─────────────────────────────────────────────────────────────────────
let R=[],cfgData=null,ckList=[],selCk=null,chatLoaded=false,firstSeen=null;
let sysData={cpu:[],ram:[]};

function age(ts){
  if(!n(ts)) return 'no data';
  const s=Math.max(0,Math.round(Date.now()/1000-ts));
  return s<60?`${s}s ago`:`${Math.floor(s/60)}m ${s%60}s ago`;
}
function dur(s){
  if(!s) return '—';
  const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),ss=s%60;
  return h?`${h}h ${m}m`:m?`${m}m ${ss}s`:`${ss}s`;
}
function barColor(p){return p<60?'#22c55e':p<80?'#f59e0b':'#ef4444'}
function setBar(id,pct){
  const el=document.getElementById(id); if(!el) return;
  el.style.width=Math.min(100,pct).toFixed(1)+'%';
  el.style.background=barColor(pct);
}

// ── Overview ──────────────────────────────────────────────────────────────────
function renderOverview(rows){
  const lr=rows[rows.length-1]||{};
  const maxS=last(rows,'run/max_steps');
  const step=n(lr.step)?lr.step:0;
  const pct=n(maxS)?Math.min(100,(step/maxS)*100):0;
  const tok=last(rows,'train/tokens_per_sec');

  document.getElementById('p-step').textContent=`step ${step.toLocaleString()} / ${n(maxS)?maxS.toLocaleString():'—'}`;
  document.getElementById('p-pct').textContent=pct.toFixed(1)+'%';
  document.getElementById('p-bar').style.width=pct.toFixed(2)+'%';
  document.getElementById('p-tok').textContent=n(tok)?Math.round(tok).toLocaleString()+' tok/s':'—';
  if(!firstSeen&&rows.length) firstSeen=rows[0].time;
  if(firstSeen) document.getElementById('p-elapsed').textContent='elapsed '+dur(Math.floor(Date.now()/1000-firstSeen));

  const tl=last(rows,'train/loss'),vl=last(rows,'val/loss');
  const ppl=last(rows,'val/perplexity');
  const gnB=last(rows,'train/grad_norm_before_clip'),gnA=last(rows,'train/grad_norm_after_clip');
  const lrv=last(rows,'train/lr');

  document.getElementById('s-tl').textContent=fmt(tl);
  document.getElementById('s-vl').textContent=fmt(vl);
  document.getElementById('s-ppl').textContent=fmt(ppl,2);
  document.getElementById('s-tok').textContent=n(tok)?Math.round(tok).toLocaleString():'—';
  document.getElementById('s-gn').textContent=(n(gnB)||n(gnA))?`${fmt(gnB,2)} → ${fmt(gnA,2)}`:'—';
  document.getElementById('s-lr').textContent=n(lrv)?lrv.toExponential(2):'—';
  document.getElementById('tb-step').textContent=`step ${step.toLocaleString()} / ${n(maxS)?maxS.toLocaleString():'—'}`;

  const running=rows.length&&lr.event!=='done';
  document.getElementById('sb-dot').className='dot'+(rows.length?(running?' run':' done'):'');
  document.getElementById('sb-st').textContent=!rows.length?'Waiting':(running?'Running':'Done');
  document.getElementById('sb-age').textContent=age(lr.time);
  document.getElementById('sb-run').textContent=window._run||'—';

  draw('ov-loss',[{color:C.a,data:pts(rows,'train/loss')},{color:C.amber,data:pts(rows,'val/loss')}]);
  draw('ov-tok',[{color:C.teal,data:pts(rows,'train/tokens_per_sec')}]);

  const tb=document.getElementById('ov-ev');
  tb.innerHTML=[...rows].reverse().slice(0,16).map(r=>{
    const ev=r.event||'train';
    const cls=ev==='eval'?'t-ev':ev==='done'?'t-dn':ev==='sample'?'t-sp':ev==='health'?'t-he':ev==='warning'?'t-wa':'t-tr';
    return `<tr>
      <td>${n(r.step)?r.step.toLocaleString():'—'}</td>
      <td><span class="tag ${cls}">${ev}</span></td>
      <td>${fmt(r['train/loss'])}</td>
      <td>${n(r['train/lr'])?r['train/lr'].toExponential(2):'—'}</td>
      <td style="color:var(--ink3);font-size:10.5px;max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${
        Object.entries(r).filter(([k])=>!['time','event','step'].includes(k)).filter(([,v])=>n(v)).slice(0,5)
          .map(([k,v])=>`${k.split('/').pop()}: ${fmt(v)}`).join(' · ')}</td>
    </tr>`;
  }).join('');
  renderAlerts(rows);
}

// ── Alerts ────────────────────────────────────────────────────────────────────
function renderAlerts(rows){
  const als=[];
  const gnB=last(rows,'train/grad_norm_before_clip');
  const tl=last(rows,'train/loss'),vl=last(rows,'val/loss');
  if(n(tl)&&!isFinite(tl)) als.push({t:'d',i:'✕',m:'Train loss is NaN/Inf — training diverged. Check LR and data.'});
  if(gnB&&gnB>10) als.push({t:'d',i:'⚡',m:`Gradient explosion: ${gnB.toFixed(2)} (threshold 10). Lower LR or increase grad_clip.`});
  else if(gnB&&gnB>5) als.push({t:'w',i:'△',m:`Elevated grad norm: ${gnB.toFixed(2)}. Watch for instability.`});
  if(n(tl)&&n(vl)&&vl-tl>0.5) als.push({t:'w',i:'△',m:`Large train/val gap (${(vl-tl).toFixed(3)}). Possible overfitting.`});
  // Check health events in recent rows
  const last10=[...rows].slice(-20).filter(r=>r.event==='warning'||r.event==='health');
  for(const r of last10){
    const msg=r['health/message']||r['message'];
    if(msg) als.push({t:'w',i:'◎',m:msg});
  }
  const hz=document.getElementById('health-alerts');
  const az=document.getElementById('alert-zone');
  if(!als.length){
    hz.innerHTML='<div class="al al-ok"><span class="al-ico">✓</span>All metrics within normal ranges.</div>';
    az.innerHTML='';return;
  }
  const mk=a=>`<div class="al al-${a.t==='d'?'d':'w'}"><span class="al-ico">${a.i}</span>${a.m}</div>`;
  hz.innerHTML=als.map(mk).join('');
  az.innerHTML=mk(als[0]);
  az.style.marginBottom='16px';
}

// ── All charts ─────────────────────────────────────────────────────────────────
function drawAll(rows){
  draw('ch-loss',[{color:C.a,data:pts(rows,'train/loss')},{color:C.amber,data:pts(rows,'val/loss')}]);
  draw('ch-gn',[{color:C.red,data:pts(rows,'train/grad_norm_before_clip')},{color:C.teal,data:pts(rows,'train/grad_norm_after_clip')}]);
  draw('ch-tput',[{color:C.purple,data:pts(rows,'train/tokens_per_sec')}]);
  draw('ch-lr',[{color:C.green,data:pts(rows,'train/lr')}]);
  draw('ch-ppl',[{color:C.amber,data:pts(rows,'val/perplexity')}]);
  draw('ch-cpu',[{color:C.a,data:sysData.cpu.map((v,i)=>({x:i,y:v}))}]);
  draw('ch-ram',[{color:C.teal,data:sysData.ram.map((v,i)=>({x:i,y:v}))}]);
  draw('h-gn',[{color:C.red,data:pts(rows,'train/grad_norm_before_clip')}]);
}

// ── System ─────────────────────────────────────────────────────────────────────
function drawSysFull(){
  draw('sy-cpu-ch',[{color:C.a,data:sysData.cpu.map((v,i)=>({x:i,y:v}))}]);
  draw('sy-ram-ch',[{color:C.teal,data:sysData.ram.map((v,i)=>({x:i,y:v}))}]);
  draw('ch-cpu',[{color:C.a,data:sysData.cpu.map((v,i)=>({x:i,y:v}))}]);
  draw('ch-ram',[{color:C.teal,data:sysData.ram.map((v,i)=>({x:i,y:v}))}]);
}

// ── Events log ─────────────────────────────────────────────────────────────────
function renderEvents(rows){
  const tb=document.getElementById('ev-body');
  tb.innerHTML=[...rows].reverse().map(r=>{
    const ev=r.event||'train';
    const cls=ev==='eval'?'t-ev':ev==='done'?'t-dn':ev==='sample'?'t-sp':ev==='health'?'t-he':ev==='warning'?'t-wa':'t-tr';
    return `<tr>
      <td>${n(r.step)?r.step.toLocaleString():'—'}</td>
      <td style="color:var(--ink3)">${r.time?new Date(r.time*1000).toLocaleTimeString():'—'}</td>
      <td><span class="tag ${cls}">${ev}</span></td>
      <td>${fmt(r['train/loss'])}</td>
      <td>${fmt(r['val/loss'])}</td>
      <td>${n(r['train/lr'])?r['train/lr'].toExponential(2):'—'}</td>
      <td>${fmt(r['train/grad_norm_before_clip'],2)}</td>
    </tr>`;
  }).join('');
}

// ── Checkpoints ────────────────────────────────────────────────────────────────
function renderCkpts(){
  const el=document.getElementById('ck-list');
  if(!ckList.length){
    el.innerHTML='<div class="empty"><div class="empty-ico">◧</div><div class="empty-txt">No checkpoints found</div><div>Checkpoints appear once training saves them</div></div>';
    document.getElementById('ck-panel').className='ck-detail';return;
  }
  document.getElementById('ck-n').textContent=ckList.length;
  el.innerHTML=`<div class="ck-grid">${ckList.map((c,i)=>`
    <div class="ck-row${selCk===i?' sel':''}" onclick="selCkpt(${i})">
      <div class="ck-step">step ${n(c.step)?c.step.toLocaleString():'?'}</div>
      <div class="ck-meta">val_loss: <b>${n(c.val_loss)?c.val_loss.toFixed(4):'—'}</b> · ${c.name}</div>
      <span class="ck-badge">v${c.schema_version||'?'}</span>
    </div>`).join('')}</div>`;
  if(selCk!==null) showCkDetail(ckList[selCk]);
}
function selCkpt(i){selCk=i;renderCkpts()}
function showCkDetail(c){
  const p=document.getElementById('ck-panel');
  p.className='ck-detail on';
  p.innerHTML=`<div class="ck-dgrid">${[
    ['File',c.name],['Step',n(c.step)?c.step.toLocaleString():'—'],
    ['Val Loss',n(c.val_loss)?c.val_loss.toFixed(6):'—'],
    ['Schema',c.schema_version||'—'],
    ['Hash',c.hash?(c.hash.slice(0,20)+'…'):'—'],
  ].map(([k,v])=>`<div class="kv"><span class="kk">${k}</span><span class="kv2">${v}</span></div>`).join('')}</div>`;
}
function syncChatSel(){
  const s=document.getElementById('chat-ck');
  const prev=s.value;
  s.innerHTML='<option value="">— select checkpoint —</option>'+
    ckList.map(c=>`<option value="${c.path}">${c.name} (step ${n(c.step)?c.step.toLocaleString():'?'}, val ${n(c.val_loss)?c.val_loss.toFixed(4):'—'})</option>`).join('');
  if(prev) s.value=prev;
}

// ── Config ─────────────────────────────────────────────────────────────────────
function renderCfg(){
  const b=document.getElementById('cfg-body');
  if(!cfgData||!Object.keys(cfgData).length){
    b.innerHTML='<div class="empty"><div class="empty-ico">⊞</div><div class="empty-txt">No config data yet</div></div>';return;
  }
  b.innerHTML=Object.entries(cfgData).map(([sec,vals])=>`
    <div class="cfg-sec">
      <div class="cfg-sec-title">${sec}</div>
      <div class="cfg-grid">
        ${Object.entries(vals).map(([k,v])=>`
          <div class="cfg-row"><span class="cfg-k">${k}</span><span class="cfg-v">${JSON.stringify(v)}</span></div>`).join('')}
      </div>
    </div>`).join('');
}

// ── Chat ───────────────────────────────────────────────────────────────────────
function addMsg(role,text){
  const box=document.getElementById('chat-msgs');
  const d=document.createElement('div');
  d.className=`msg msg-${role}`;
  d.textContent=text;
  box.appendChild(d);
  box.scrollTop=box.scrollHeight;
  return d;
}
function clearChat(){
  document.getElementById('chat-msgs').innerHTML='<div class="msg msg-s">Chat cleared.</div>';
  chatLoaded=false;
  document.getElementById('chat-st').textContent='No checkpoint loaded';
}
function chatKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat()}}

async function loadCkpt(){
  const path=document.getElementById('chat-ck').value;
  if(!path){document.getElementById('chat-st').textContent='Select a checkpoint first';return;}
  document.getElementById('chat-st').textContent='Loading…';
  try{
    const r=await fetch('/api/chat/load',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path})
    });
    if(!r.ok){
      const txt=await r.text();
      document.getElementById('chat-st').textContent=`HTTP ${r.status}: ${txt.slice(0,120)}`;
      return;
    }
    const d=await r.json();
    if(d.ok){
      chatLoaded=true;
      document.getElementById('chat-st').textContent='✓ '+path.split(/[\\/]/).pop();
      addMsg('s','✓ Checkpoint loaded. Start chatting!');
    } else {
      document.getElementById('chat-st').textContent='Error: '+(d.error||'unknown');
      addMsg('s','Load error: '+(d.error||'unknown'));
    }
  } catch(e){
    document.getElementById('chat-st').textContent='Network error — is nanoforge serve running?';
  }
}

async function sendChat(){
  const inp=document.getElementById('chat-in');
  const text=inp.value.trim();
  if(!text) return;
  if(!chatLoaded){addMsg('s','Load a checkpoint first.');return;}
  inp.value='';
  addMsg('u',text);
  const ph=addMsg('m','▋');
  document.getElementById('chat-btn').disabled=true;
  try{
    const payload={
      prompt:text,
      temperature:Math.max(0,Math.min(2,parseFloat(document.getElementById('cfg-t').value)||0.8)),
      top_k:Math.max(1,parseInt(document.getElementById('cfg-k').value,10)||50),
      max_new_tokens:Math.max(8,parseInt(document.getElementById('cfg-m').value,10)||256),
      mode:document.getElementById('cfg-md').value||'balanced',
    };
    const r=await fetch('/api/chat/generate',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    if(!r.ok){
      const txt=await r.text();
      ph.textContent=`[HTTP ${r.status}] ${txt.slice(0,200)}`;
    } else {
      const d=await r.json();
      ph.textContent=d.text||'(empty response)';
    }
  } catch(e){ph.textContent='[network error generating response]';}
  document.getElementById('chat-btn').disabled=false;
}

// ── Polling ────────────────────────────────────────────────────────────────────
async function refreshMetrics(){
  try{
    const r=await fetch('/api/metrics');
    if(!r.ok) return;
    const p=await r.json();
    R=p.rows||[];
    window._run=p.run;
    renderOverview(R);
    renderEvents(R);
    if(document.getElementById('pg-charts').classList.contains('on')) drawAll(R);
    if(document.getElementById('pg-health').classList.contains('on')) drawAll(R);
    if(p.checkpoints){
      ckList=p.checkpoints;
      document.getElementById('ck-n').textContent=ckList.length;
      syncChatSel();
      if(document.getElementById('pg-checkpoints').classList.contains('on')) renderCkpts();
    }
    if(p.config) cfgData=p.config;
  } catch(_){}
}

async function refreshSystem(){
  try{
    const r=await fetch('/api/system'); if(!r.ok) return;
    const d=await r.json();
    if(d.cpu_history) sysData.cpu=d.cpu_history;
    if(d.ram_history) sysData.ram=d.ram_history;
    if(n(d.cpu_pct)){document.getElementById('sy-cpu').textContent=d.cpu_pct.toFixed(1)+'%';setBar('sb-cpu',d.cpu_pct)}
    if(n(d.ram_used)&&n(d.ram_total)){
      document.getElementById('sy-ram').textContent=d.ram_used.toFixed(1)+' GB';
      document.getElementById('sy-ram-total').textContent=`${d.ram_used.toFixed(1)} / ${d.ram_total.toFixed(1)} GB`;
      setBar('sb-ram',(d.ram_used/d.ram_total)*100);
    }
    if(n(d.cpu_temp)){document.getElementById('sy-temp').textContent=d.cpu_temp.toFixed(0)+' °C';setBar('sb-temp',Math.min(100,(d.cpu_temp/100)*100))}
    else document.getElementById('sy-temp').textContent='N/A';
    if(n(d.cpu_freq_ghz)) document.getElementById('sy-freq').textContent=d.cpu_freq_ghz.toFixed(2)+' GHz';
    if(d.cpu_logical&&d.cpu_physical) document.getElementById('sy-cores').textContent=`${d.cpu_logical} / ${d.cpu_physical}`;
    if(n(d.swap_used)&&n(d.swap_total)&&d.swap_total>0){
      document.getElementById('sy-swap').textContent=`${d.swap_used.toFixed(1)} GB`;
      setBar('sb-swap',(d.swap_used/d.swap_total)*100);
    }
    drawSysFull();
  } catch(_){}
}

refreshMetrics().catch(console.error);
refreshSystem().catch(console.error);
setInterval(()=>refreshMetrics().catch(console.error),1500);
setInterval(()=>refreshSystem().catch(console.error),2000);
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scan_checkpoints(run_path: Path) -> list[dict]:
    """Return checkpoint metadata dicts sorted newest-first."""
    import json as _json
    ckpts = []
    for pt in sorted(run_path.glob("*.pt"), reverse=True):
        meta_path = pt.with_suffix(".pt.meta.json")
        entry: dict = {
            "name": pt.name,
            "path": str(pt),
            "schema_version": None,
            "step": None,
            "val_loss": None,
            "hash": None,
        }
        if meta_path.exists():
            try:
                m = _json.loads(meta_path.read_text(encoding="utf-8"))
                for k in ("schema_version", "step", "val_loss", "hash"):
                    entry[k] = m.get(k)
            except Exception:
                pass
        ckpts.append(entry)
    return ckpts


def _flatten_config(cfg) -> dict:
    """Turn NanoforgeConfig dataclass or plain dict into {section: {key: val}}."""
    import dataclasses
    if cfg is None:
        return {}
    if dataclasses.is_dataclass(cfg):
        raw = dataclasses.asdict(cfg)
    elif isinstance(cfg, dict):
        raw = cfg
    else:
        return {}
    result: dict = {}
    for section, vals in raw.items():
        if isinstance(vals, dict):
            result[section] = dict(vals)
        else:
            result.setdefault("misc", {})[section] = vals
    return result


def _load_engine_compat(path: str):
    """
    Load GenerationEngine from a checkpoint, handling both:
      - new NanoforgeConfig checkpoints (schema_version >= 2)
      - old dict-config checkpoints (schema_version 1) with graceful fallback
    Returns (engine, error_string).  One of them will be None.
    """
    try:
        from nanoforge.generation.engine import GenerationEngine
        engine = GenerationEngine.from_checkpoint(path)
        return engine, None
    except RuntimeError as exc:
        msg = str(exc)
        if "dict config" in msg or "dict" in msg.lower():
            # Old checkpoint with raw dict config — try manual reconstruction
            try:
                import torch
                from nanoforge.training.checkpoint import load_checkpoint
                from nanoforge.config import NanoforgeConfig, ModelConfig, TrainConfig, DataConfig, InferenceConfig
                from nanoforge.model.transformer import NanoforgeForCausalLM
                from nanoforge.data.tokenizer import load_tokenizer
                from nanoforge.training.utils import resolve_device
                from nanoforge.generation.engine import GenerationEngine

                payload = load_checkpoint(path, map_location="cpu")
                raw_cfg = payload["config"]
                if isinstance(raw_cfg, dict):
                    # Reconstruct config from dict
                    model_d = raw_cfg.get("model", {})
                    train_d = raw_cfg.get("training", {})
                    data_d  = raw_cfg.get("data", {})
                    inf_d   = raw_cfg.get("inference", {})
                    cfg = NanoforgeConfig(
                        model=ModelConfig(**{k: v for k, v in model_d.items() if k in ModelConfig.__dataclass_fields__}),
                        training=TrainConfig(**{k: v for k, v in train_d.items() if k in TrainConfig.__dataclass_fields__}),
                        data=DataConfig(**{k: v for k, v in data_d.items() if k in DataConfig.__dataclass_fields__}),
                        inference=InferenceConfig(**{k: v for k, v in inf_d.items() if k in InferenceConfig.__dataclass_fields__}),
                    )
                else:
                    cfg = raw_cfg

                model = NanoforgeForCausalLM(cfg.model)
                model.load_state_dict(payload["model"], strict=True)
                tok_type = cfg.data.tokenizer_type
                tok_path = cfg.data.tokenizer_path
                tokenizer = load_tokenizer(tok_type, tok_path)
                device = resolve_device("auto")
                engine = GenerationEngine(model, tokenizer, device=device)
                return engine, None
            except Exception as inner:
                return None, f"Old dict-config checkpoint load failed: {inner}"
        return None, msg
    except Exception as exc:
        return None, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app factory
# ──────────────────────────────────────────────────────────────────────────────

def create_dashboard_app(run_dir):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel
    except Exception as exc:
        raise RuntimeError(
            "Install serve extras to use dashboard: pip install -e .[serve]"
        ) from exc

    import json as _json

    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.jsonl"
    config_path  = run_path / "config.yaml"

    app = FastAPI(title="Nanoforge Dashboard")

    _engine_holder: dict = {"engine": None, "path": None}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/metrics")
    def metrics(limit: int = 2000):
        rows = read_jsonl_tail(metrics_path, limit=limit)
        checkpoints = _scan_checkpoints(run_path)

        cfg_data: dict = {}
        if config_path.exists():
            try:
                from nanoforge.config import load_config
                cfg_data = _flatten_config(load_config(config_path))
            except Exception:
                pass

        # Fallback: try the first checkpoint's YAML sidecar
        if not cfg_data and checkpoints:
            for ck in checkpoints:
                sidecar = Path(ck["path"]).with_suffix(".yaml")
                if sidecar.exists():
                    try:
                        from nanoforge.config import load_config
                        cfg_data = _flatten_config(load_config(sidecar))
                        break
                    except Exception:
                        pass

        return json_safe({
            "run": run_path.name,
            "path": str(metrics_path),
            "rows": rows,
            "checkpoints": checkpoints,
            "config": cfg_data,
        })

    @app.get("/api/system")
    def system_stats():
        try:
            import psutil
        except ImportError:
            return {"error": "psutil not installed — run: pip install psutil"}

        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        freq = psutil.cpu_freq()

        cpu_pct  = psutil.cpu_percent(interval=0.2)
        ram_used = round(vm.used  / 1e9, 3)

        # Accumulate server-side history
        _sys_cpu_history.append(cpu_pct)
        _sys_ram_history.append(ram_used)

        data: dict = {
            "cpu_pct":      cpu_pct,
            "cpu_logical":  psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "cpu_freq_ghz": round(freq.current / 1000, 3) if freq else None,
            "ram_used":     ram_used,
            "ram_total":    round(vm.total / 1e9, 3),
            "swap_used":    round(sw.used  / 1e9, 3),
            "swap_total":   round(sw.total / 1e9, 3),
            "cpu_temp":     None,
            # Send full accumulated history so browser always has complete charts
            "cpu_history":  list(_sys_cpu_history),
            "ram_history":  list(_sys_ram_history),
        }

        # CPU temperature — Linux
        try:
            temps = psutil.sensors_temperatures()
            for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                if key in temps and temps[key]:
                    data["cpu_temp"] = round(temps[key][0].current, 1)
                    break
        except Exception:
            pass

        # CPU temperature — Windows via WMI + OpenHardwareMonitor
        if data["cpu_temp"] is None:
            try:
                import wmi
                w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                for s in w.Sensor():
                    if s.SensorType == "Temperature" and "CPU" in s.Name:
                        data["cpu_temp"] = round(float(s.Value), 1)
                        break
            except Exception:
                pass

        return data

    # ── Chat ─────────────────────────────────────────────────────────────────

    class LoadRequest(BaseModel):
        path: str

    class GenerateRequest(BaseModel):
        prompt: str
        temperature: float = 0.8
        top_k: int = 50
        max_new_tokens: int = 256
        mode: str = "balanced"

    @app.post("/api/chat/load")
    def chat_load(req: LoadRequest):
        engine, err = _load_engine_compat(req.path)
        if err:
            return {"ok": False, "error": err}
        _engine_holder["engine"] = engine
        _engine_holder["path"] = req.path
        return {"ok": True}

    @app.post("/api/chat/generate")
    def chat_generate(req: GenerateRequest):
        engine = _engine_holder.get("engine")
        if engine is None:
            return {"text": "[no checkpoint loaded — use Load Checkpoint first]"}
        try:
            from nanoforge.generation.sampling import SamplingConfig
            sc = SamplingConfig(
                mode=req.mode,
                temperature=req.temperature,
                top_k=req.top_k,
            )
            result = engine.complete(req.prompt, max_new_tokens=req.max_new_tokens, sampling=sc)
            return {"text": result or "(empty response)"}
        except Exception as exc:
            return {"text": f"[generation error: {exc}]"}

    @app.get("/favicon.ico")
    def favicon():
        return HTMLResponse("", status_code=204)

    return app


def serve_dashboard(run_dir, host: str = "127.0.0.1", port: int = 7860):
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install serve extras: pip install -e .[serve]") from exc
    uvicorn.run(create_dashboard_app(run_dir), host=host, port=port)