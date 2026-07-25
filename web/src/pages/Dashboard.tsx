import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { GitCommit } from 'lucide-react'
import { get } from '../lib/api'
import { Card, CardTitle, PageHeader, Empty } from '../lib/ui'

function StatCard({ label, value, color }: { label: string; value: string | undefined; color: string }) {
  return (
    <div className="card" style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ fontSize: 11, color: 'var(--color-ink-3)', fontWeight: 500, letterSpacing: '0.04em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, fontFamily: 'var(--font-mono)', color: value ? color : 'var(--color-ink-3)', letterSpacing: '-0.03em' }}>
        {value ?? '—'}
      </div>
    </div>
  )
}

function MetricChart({ data, dataKey, color, label }: { data: any[]; dataKey: string; color: string; label: string }) {
  return (
    <Card>
      <CardTitle>{label}</CardTitle>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} margin={{ top: 2, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-line)" />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} />
          <YAxis tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} width={52} />
          <Tooltip
            contentStyle={{ background: 'var(--color-surface-3)', border: '1px solid var(--color-line-2)', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: 'var(--color-ink-2)' }}
          />
          <Line type="monotone" dataKey={dataKey} stroke={color} dot={false} strokeWidth={1.75} />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}

// Normalize namespaced metric keys (train/loss → loss, val/loss → val_loss, etc.)
function normalizeRow(row: any): any {
  const out: any = { step: row.step, time: row.time, event: row.event }
  for (const [k, v] of Object.entries(row)) {
    if (k.startsWith('train/')) out[k.slice(6)] = v
    else if (k.startsWith('val/')) out['val_' + k.slice(4)] = v
    else out[k] = v
  }
  return out
}

export default function Dashboard() {
  const { data, isLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => get('/api/metrics'),
    refetchInterval: 3000,
  })

  const raw: any[] = data?.rows ?? []
  const rows = raw.map(normalizeRow)
  const last = rows[rows.length - 1]
  const ckpts: any[] = data?.checkpoints ?? []
  const valRows = rows.filter(r => r.val_loss != null)

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
      />

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
        <StatCard label="Step" value={last?.step?.toLocaleString()} color="var(--color-accent)" />
        <StatCard label="Train loss" value={last?.loss?.toFixed(4)} color="var(--color-neg)" />
        <StatCard label="Val loss" value={last?.val_loss?.toFixed(4)} color="var(--color-warn)" />
        <StatCard label="Tok/s" value={last?.tokens_per_sec?.toFixed(0)} color="var(--color-pos)" />
      </div>

      {/* Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
        <MetricChart data={rows.slice(-300)} dataKey="loss" color="var(--color-neg)" label="Train loss" />
        <MetricChart data={valRows.slice(-100)} dataKey="val_loss" color="var(--color-warn)" label="Val loss" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <MetricChart data={rows.slice(-300)} dataKey="grad_norm" color="var(--color-accent)" label="Grad norm" />

        {/* Checkpoints */}
        <Card>
          <CardTitle right={
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-ink-3)' }}>
              {ckpts.length} saved
            </span>
          }>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><GitCommit size={13} /> Checkpoints</span>
          </CardTitle>
          {ckpts.length === 0
            ? <Empty>No checkpoints yet.</Empty>
            : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {ckpts.slice(0, 8).map((c: any) => (
                  <div key={c.path} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '7px 10px', background: 'var(--color-surface-3)', borderRadius: 8,
                  }}>
                    <span className="mono" style={{ fontSize: 12, color: 'var(--color-ink-2)' }}>{c.name}</span>
                    <span style={{ fontSize: 11, color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)' }}>
                      {c.step != null ? `step ${c.step}` : '—'}
                      {c.val_loss != null ? ` · ${c.val_loss.toFixed(4)}` : ''}
                    </span>
                  </div>
                ))}
              </div>
            )
          }
        </Card>
      </div>
    </div>
  )
}
