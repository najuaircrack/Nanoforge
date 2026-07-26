import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { GitCommit, Activity, Layers } from 'lucide-react'
import { get } from '../lib/api'
import { Card, CardTitle, PageHeader, Empty, Gauge, Stat, ArchFlow } from '../lib/ui'
import { normalizeRows, fmt, fmtInt, fmtCompact, fmtPct, fmtDur, fmtExp } from '../lib/metrics'

const CHART_TOOLTIP = {
  contentStyle: { background: 'var(--color-surface-3)', border: '1px solid var(--color-line-2)', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: 'var(--color-ink-2)' },
}

function AreaMetric({ data, dataKey, color, label, right }: { data: any[]; dataKey: string; color: string; label: string; right?: React.ReactNode }) {
  const gid = 'a-' + dataKey.replace(/\W/g, '')
  return (
    <Card>
      <CardTitle right={right}>{label}</CardTitle>
      <ResponsiveContainer width="100%" height={150}>
        <AreaChart data={data} margin={{ top: 2, right: 4, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.35} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-line)" vertical={false} />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} />
          <YAxis tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} width={50} />
          <Tooltip {...CHART_TOOLTIP} />
          <Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#${gid})`} strokeWidth={1.75} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  )
}

export default function Dashboard() {
  const [runRoot] = useState('runs')
  const { data: runsData } = useQuery({ queryKey: ['runs', runRoot], queryFn: () => get('/api/runs', { root: runRoot }), refetchInterval: 8000 })
  const runs: any[] = runsData?.runs ?? []

  const { data, isLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => get('/api/metrics'),
    refetchInterval: 3000,
  })
  const { data: summaryData } = useQuery({
    queryKey: ['metrics-summary'],
    queryFn: () => get('/api/metrics/summary'),
    refetchInterval: 3000,
  })

  const rows = normalizeRows(data?.rows ?? [])
  const s = summaryData?.summary ?? {}
  const ckpts: any[] = data?.checkpoints ?? []
  const cfg = data?.config ?? {}
  const valRows = rows.filter(r => r.val_loss != null)

  const running = s.progress != null && s.progress < 1

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Dashboard"
        subtitle={
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {data?.run ? `run: ${data.run}` : 'no active run'}
            {isLoading && <span className="live-dot" style={{ color: 'var(--color-accent)', fontSize: 10 }}>●</span>}
          </span>
        }
        right={running
          ? <span className="chip" style={{ color: 'var(--color-pos)' }}><span className="live-dot">●</span> training</span>
          : <span className="chip">idle</span>}
      />

      {/* Gauges */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
        <Card style={{ display: 'flex', justifyContent: 'center', padding: 14 }}>
          <Gauge label="Progress" value={s.progress ?? 0} max={1}
            display={s.progress != null ? fmtPct(s.progress, 0) : '—'} color="var(--color-accent)" />
        </Card>
        <Card style={{ display: 'flex', justifyContent: 'center', padding: 14 }}>
          <Gauge label="Throughput" value={s.tokens_per_sec ?? 0} max={Math.max(s.mean_tokens_per_sec * 1.5 || 1, s.tokens_per_sec || 1)}
            display={fmtCompact(s.tokens_per_sec)} unit="tok/s" color="var(--color-pos)" />
        </Card>
        <Card style={{ display: 'flex', justifyContent: 'center', padding: 14 }}>
          <Gauge label="Grad norm" value={Math.min(s.grad_norm ?? 0, 5)} max={5}
            display={fmt(s.grad_norm, 2)} color="var(--color-warn)" />
        </Card>
        <Card style={{ display: 'flex', justifyContent: 'center', padding: 14 }}>
          <Gauge label="Loss" value={s.train_loss != null ? Math.max(0, 8 - s.train_loss) : 0} max={8}
            display={fmt(s.train_loss, 3)} color="var(--color-neg)" />
        </Card>
      </div>

      {/* Stat strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12, marginBottom: 16 }}>
        <Stat label="Step" value={fmtInt(s.step)} sub={s.max_steps ? `of ${fmtInt(s.max_steps)}` : undefined} color="var(--color-accent)" />
        <Stat label="Val loss" value={fmt(s.val_loss, 4)} sub={s.best_val_loss != null ? `best ${fmt(s.best_val_loss, 4)}` : undefined} color="var(--color-warn)" />
        <Stat label="Perplexity" value={fmt(s.perplexity, 2)} />
        <Stat label="LR" value={fmtExp(s.lr)} />
        <Stat label="Tokens" value={fmtCompact(s.tokens_seen)} sub="seen" color="var(--color-pos)" />
        <Stat label="ETA" value={fmtDur(s.eta_sec)} sub={s.elapsed_sec ? `elapsed ${fmtDur(s.elapsed_sec)}` : undefined} />
      </div>

      {/* Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
        <AreaMetric data={rows.slice(-300)} dataKey="loss" color="var(--color-neg)" label="Train loss" />
        <AreaMetric data={valRows.slice(-120)} dataKey="val_loss" color="var(--color-warn)" label="Val loss" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
        <AreaMetric data={rows.slice(-300)} dataKey="tokens_per_sec" color="var(--color-pos)" label="Throughput (tok/s)" />
        <Card>
          <CardTitle>Grad norm & entropy</CardTitle>
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={rows.slice(-300)} margin={{ top: 2, right: 4, left: -18, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-line)" vertical={false} />
              <XAxis dataKey="step" tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} width={50} />
              <Tooltip {...CHART_TOOLTIP} />
              <Line type="monotone" dataKey="grad_norm" stroke="var(--color-accent)" dot={false} strokeWidth={1.6} />
              <Line type="monotone" dataKey="health_logit_entropy" stroke="var(--color-pos)" dot={false} strokeWidth={1.6} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Architecture + checkpoints + runs */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        <Card>
          <CardTitle><span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Layers size={13} /> Architecture</span></CardTitle>
          {cfg?.model ? <ArchFlow cfg={cfg} /> : <Empty>No config loaded for this run.</Empty>}
        </Card>

        <Card>
          <CardTitle right={<span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-ink-3)' }}>{ckpts.length} saved</span>}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><GitCommit size={13} /> Checkpoints</span>
          </CardTitle>
          {ckpts.length === 0 ? <Empty>No checkpoints yet.</Empty> : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {ckpts.slice(0, 9).map((c: any) => (
                <div key={c.path} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 10px', background: 'var(--color-surface-3)', borderRadius: 8 }}>
                  <span className="mono" style={{ fontSize: 12, color: 'var(--color-ink-2)' }}>{c.name}</span>
                  <span style={{ fontSize: 11, color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)' }}>
                    {c.step != null ? `step ${c.step}` : '—'}{c.val_loss != null ? ` · ${c.val_loss.toFixed(4)}` : ''}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card>
          <CardTitle right={<span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-ink-3)' }}>{runs.length}</span>}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Activity size={13} /> Runs</span>
          </CardTitle>
          {runs.length === 0 ? <Empty>No runs under runs/</Empty> : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 300, overflow: 'auto' }}>
              {runs.slice(0, 12).map((r: any) => (
                <div key={r.path} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 10px', background: r.name === data?.run ? 'var(--color-surface-3)' : 'transparent', borderRadius: 8, border: '1px solid ' + (r.name === data?.run ? 'var(--color-line-2)' : 'transparent') }}>
                  <span className="mono" style={{ fontSize: 12, color: 'var(--color-ink-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name}</span>
                  <span style={{ fontSize: 11, color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
                    {r.step != null ? `s${fmtInt(r.step)}` : '—'}{r.val_loss != null ? ` · ${r.val_loss.toFixed(3)}` : ''}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
