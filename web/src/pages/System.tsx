import { useQuery } from '@tanstack/react-query'
import { AreaChart, Area, ResponsiveContainer, YAxis, Tooltip, XAxis, CartesianGrid } from 'recharts'
import { get, fmtBytes } from '../lib/api'
import { Card, CardTitle, PageHeader, Empty } from '../lib/ui'

function Bar({ label, pct, detail, color }: { label: string; pct: number | undefined; detail: string; color: string }) {
  const p = Math.min(100, pct ?? 0)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
        <span style={{ color: 'var(--color-ink-2)' }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ink)' }}>
          {pct != null ? `${pct.toFixed(0)}%` : '—'}{detail ? `  ${detail}` : ''}
        </span>
      </div>
      <div style={{ height: 6, background: 'var(--color-surface-3)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${p}%`, background: color, borderRadius: 3, transition: 'width 0.4s ease' }} />
      </div>
    </div>
  )
}

export default function System() {
  const { data, dataUpdatedAt } = useQuery({
    queryKey: ['system'],
    queryFn: () => get('/api/system'),
    refetchInterval: 2000,
  })

  const cpuHist = (data?.cpu_history ?? []).map((v: number, i: number) => ({ i, v }))
  const ramHist = (data?.ram_history ?? []).map((v: number, i: number) => ({ i, v }))

  const envRows = [
    ['Platform', data?.platform],
    ['Python', data?.python],
    ['Torch', data?.torch],
    ['Device', data?.device],
    ['GPU', data?.gpu_name],
    ['CUDA', data?.cuda],
  ].filter(([, v]) => v != null)

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="System"
        subtitle={
          dataUpdatedAt
            ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-ink-3)' }}>
                updated {new Date(dataUpdatedAt).toLocaleTimeString()}
              </span>
            : undefined
        }
      />

      {data?.error && (
        <div style={{ marginBottom: 14, padding: '10px 14px', background: 'var(--color-surface-2)', border: '1px solid var(--color-line)', borderRadius: 8, color: 'var(--color-warn)', fontSize: 13 }}>
          {data.error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 14 }}>
        <Card>
          <CardTitle>CPU</CardTitle>
          <div style={{ marginBottom: 12 }}>
            <Bar label="Usage" pct={data?.cpu} detail={data?.cpu_count ? `${data.cpu_count} cores` : ''} color="var(--color-accent)" />
          </div>
          <ResponsiveContainer width="100%" height={90}>
            <AreaChart data={cpuHist} margin={{ top: 2, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-line)" />
              <XAxis dataKey="i" hide />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} width={36} />
              <Tooltip contentStyle={{ background: 'var(--color-surface-3)', border: '1px solid var(--color-line-2)', borderRadius: 8, fontSize: 11 }} />
              <Area type="monotone" dataKey="v" stroke="var(--color-accent)" fill="var(--color-accent)" fillOpacity={0.12} dot={false} strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <CardTitle>Memory</CardTitle>
          <div style={{ marginBottom: 12 }}>
            <Bar
              label="RAM"
              pct={data?.ram_percent}
              detail={data?.ram_used != null ? `${fmtBytes(data.ram_used)} / ${fmtBytes(data.ram_total)}` : ''}
              color="var(--color-pos)"
            />
          </div>
          <ResponsiveContainer width="100%" height={90}>
            <AreaChart data={ramHist} margin={{ top: 2, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-line)" />
              <XAxis dataKey="i" hide />
              <YAxis tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} width={36} />
              <Tooltip contentStyle={{ background: 'var(--color-surface-3)', border: '1px solid var(--color-line-2)', borderRadius: 8, fontSize: 11 }} />
              <Area type="monotone" dataKey="v" stroke="var(--color-pos)" fill="var(--color-pos)" fillOpacity={0.12} dot={false} strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card>
        <CardTitle>Environment</CardTitle>
        {envRows.length === 0
          ? <Empty>No environment info available.</Empty>
          : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0 24px' }}>
              {envRows.map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '5px 0', borderBottom: '1px solid var(--color-line)' }}>
                  <span style={{ color: 'var(--color-ink-2)' }}>{k}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ink)' }}>{String(v)}</span>
                </div>
              ))}
            </div>
          )
        }
      </Card>
    </div>
  )
}
