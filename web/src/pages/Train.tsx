import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Play, Square, RefreshCw } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { get, post } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Field, LogBox } from '../lib/ui'

function normalizeRow(row: any): any {
  const out: any = { step: row.step }
  for (const [k, v] of Object.entries(row)) {
    if (k.startsWith('train/')) out[k.slice(6)] = v
    else if (k.startsWith('val/')) out['val_' + k.slice(4)] = v
    else out[k] = v
  }
  return out
}

export default function Train() {
  const [config, setConfig] = useState('configs/nano-debug.yaml')
  const [mode, setMode] = useState<'train' | 'pretrain' | 'pipeline' | 'auto-train'>('train')
  const [jobId, setJobId] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [exitCode, setExitCode] = useState<number | null>(null)

  const { data: cfgList } = useQuery({ queryKey: ['cfg-templates'], queryFn: () => get('/api/config/templates') })
  const templates: string[] = cfgList?.templates ?? []

  const { data: metricsData } = useQuery({
    queryKey: ['train-metrics'],
    queryFn: () => get('/api/metrics'),
    refetchInterval: running ? 3000 : false,
    enabled: running,
  })
  const rows = (metricsData?.rows ?? []).map(normalizeRow)
  const last = rows[rows.length - 1]

  async function startJob() {
    setLogs([]); setExitCode(null); setRunning(true)
    const r = await post('/api/jobs/start', { type: mode, config })
    if (!r.ok) { setLogs([`Error: ${r.error}`]); setRunning(false); return }
    setJobId(r.job_id)
    const es = new EventSource(`/stream/job/${r.job_id}`)
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'line') setLogs(l => [...l.slice(-600), msg.line])
      else if (msg.type === 'done') { setExitCode(msg.code); setRunning(false); es.close() }
    }
    es.onerror = () => { setRunning(false); es.close() }
  }

  async function stopJob() {
    if (!jobId) return
    await post('/api/jobs/stop', { job_id: jobId })
    setRunning(false)
  }

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Train"
        right={
          exitCode !== null
            ? <span style={{ fontSize: 13, color: exitCode === 0 ? 'var(--color-pos)' : 'var(--color-neg)' }}>
                {exitCode === 0 ? '✓ Completed' : `Exit ${exitCode}`}
              </span>
            : running
              ? <span style={{ fontSize: 13, color: 'var(--color-pos)', display: 'flex', alignItems: 'center', gap: 7 }}>
                  <span className="live-dot">●</span> Training
                </span>
              : undefined
        }
      />

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 14 }}>
        {/* Config panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card>
            <CardTitle>Configuration</CardTitle>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <Field label="Mode">
                <select value={mode} onChange={e => setMode(e.target.value as any)} className="select">
                  <option value="train">train</option>
                  <option value="pretrain">pretrain</option>
                  <option value="pipeline">pipeline</option>
                  <option value="auto-train">auto-train</option>
                </select>
              </Field>
              <Field label="Config file">
                <input value={config} onChange={e => setConfig(e.target.value)} className="input" placeholder="configs/nano-debug.yaml" />
              </Field>
              {templates.length > 0 && (
                <Field label="Template">
                  <select onChange={e => e.target.value && setConfig(e.target.value)} className="select" defaultValue="">
                    <option value="">Pick a template...</option>
                    {templates.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </Field>
              )}
              <div style={{ display: 'flex', gap: 8, paddingTop: 4 }}>
                <Button onClick={startJob} disabled={running || !config} variant="primary" icon={<Play size={13} />} style={{ flex: 1 }}>
                  Start
                </Button>
                <Button onClick={stopJob} disabled={!running} icon={<Square size={13} />} style={{ flex: 1, color: running ? 'var(--color-neg)' : undefined }}>
                  Stop
                </Button>
              </div>
            </div>
          </Card>

          {/* Live metrics */}
          {last && (
            <Card>
              <CardTitle>Live metrics</CardTitle>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                {[
                  ['Step', last?.step?.toLocaleString()],
                  ['Loss', last?.loss?.toFixed(4)],
                  ['Val loss', last?.val_loss?.toFixed(4)],
                  ['LR', last?.lr?.toExponential(2)],
                  ['Grad norm', last?.grad_norm?.toFixed(3)],
                  ['Tok/s', last?.tokens_per_sec?.toFixed(0)],
                ].map(([k, v]) => v != null ? (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '5px 0', borderBottom: '1px solid var(--color-line)' }}>
                    <span style={{ color: 'var(--color-ink-2)' }}>{k}</span>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>{v}</span>
                  </div>
                ) : null)}
              </div>
            </Card>
          )}
        </div>

        {/* Charts + log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {rows.length > 0 && (
            <Card>
              <CardTitle>Loss curve</CardTitle>
              <ResponsiveContainer width="100%" height={130}>
                <LineChart data={rows.slice(-300)} margin={{ top: 2, right: 4, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-line)" />
                  <XAxis dataKey="step" tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--color-ink-3)' }} tickLine={false} axisLine={false} width={52} />
                  <Tooltip contentStyle={{ background: 'var(--color-surface-3)', border: '1px solid var(--color-line-2)', borderRadius: 8, fontSize: 12 }} />
                  <Line type="monotone" dataKey="loss" stroke="var(--color-neg)" dot={false} strokeWidth={1.75} />
                  <Line type="monotone" dataKey="val_loss" stroke="var(--color-warn)" dot={false} strokeWidth={1.75} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          <Card style={{ flex: 1 }}>
            <CardTitle right={
              <Button variant="ghost" onClick={() => setLogs([])} icon={<RefreshCw size={11} />} style={{ fontSize: 11, padding: '3px 8px' }}>
                Clear
              </Button>
            }>
              Output log
            </CardTitle>
            <LogBox lines={logs} running={running} />
          </Card>
        </div>
      </div>
    </div>
  )
}
