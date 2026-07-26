/* Shared metric normalization + formatting.
   The trainer writes namespaced JSONL keys (train/loss, val/loss, health/…);
   the web strips the namespace into flat, chart-friendly field names. */

export type Row = Record<string, any>

// train/loss → loss, val/loss → val_loss, health/x → health_x, run/x → run_x
export function normalizeRow(row: Row): Row {
  const out: Row = { step: row.step, time: row.time, event: row.event }
  for (const [k, v] of Object.entries(row)) {
    if (k.startsWith('train/')) out[k.slice(6)] = v
    else if (k.startsWith('val/')) out['val_' + k.slice(4)] = v
    else if (k.startsWith('health/')) out['health_' + k.slice(7)] = v
    else if (k.startsWith('run/')) out['run_' + k.slice(4)] = v
    else if (k.startsWith('sample/')) out['sample_' + k.slice(7)] = v
    else out[k] = v
  }
  return out
}

export function normalizeRows(rows: Row[]): Row[] {
  return rows.map(normalizeRow)
}

// Fold rows into the last-non-null value per flattened key.
export function latestOf(rows: Row[]): Row {
  const latest: Row = {}
  for (const r of rows) {
    for (const [k, v] of Object.entries(r)) {
      if (v != null) latest[k] = v
    }
  }
  return latest
}

// ── Formatters ────────────────────────────────────────────────────────────────
export function fmt(v: any, digits = 4): string {
  if (v == null || Number.isNaN(v)) return '—'
  if (typeof v !== 'number') return String(v)
  return v.toFixed(digits)
}

export function fmtInt(v: any): string {
  if (v == null || Number.isNaN(v)) return '—'
  return Math.round(v).toLocaleString()
}

// 1234 → 1.23k, 1.2e6 → 1.20M
export function fmtCompact(v: any): string {
  if (v == null || Number.isNaN(v)) return '—'
  const n = Number(v)
  const abs = Math.abs(n)
  if (abs >= 1e12) return (n / 1e12).toFixed(2) + 'T'
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (abs >= 1e3) return (n / 1e3).toFixed(2) + 'k'
  return String(Math.round(n))
}

export function fmtPct(v: any, digits = 1): string {
  if (v == null || Number.isNaN(v)) return '—'
  return (v * 100).toFixed(digits) + '%'
}

export function fmtExp(v: any, digits = 2): string {
  if (v == null || Number.isNaN(v)) return '—'
  return Number(v).toExponential(digits)
}

// seconds → 1h 02m 03s
export function fmtDur(sec: any): string {
  if (sec == null || Number.isNaN(sec) || sec < 0) return '—'
  const s = Math.floor(sec % 60)
  const m = Math.floor((sec / 60) % 60)
  const h = Math.floor(sec / 3600)
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`
  return `${s}s`
}
