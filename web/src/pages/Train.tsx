import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Play, Square, RefreshCw } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { get, errorMessage } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Field, LogBox, StatusText } from '../lib/ui'
import { normalizeRows, latestOf, fmt } from '../lib/metrics'
import { useJob } from '../lib/useJob'
export default function Train() {
  const [config, setConfig] = useState('configs/nano-debug.yaml'), [mode, setMode] = useState('train')
  const [inputs, setInputs] = useState(''), [name, setName] = useState(''), [showLogs, setShowLogs] = useState(true)
  const monitor = useJob('training')
  const templates = useQuery({ queryKey: ['cfg-templates'], queryFn: () => get('/api/config/templates') })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: () => get('/api/jobs'), refetchInterval: 5000 })
  const metrics = useQuery({ queryKey: ['train-metrics', monitor.jobId], enabled: !!monitor.jobId,
    queryFn: ({ signal }) => get('/api/metrics', { job_id: monitor.jobId }, { signal }), refetchInterval: monitor.running ? 3000 : false })
  const rows = normalizeRows(metrics.data?.rows ?? []), latest = latestOf(rows)
  async function start() {
    setShowLogs(true)
    try { await monitor.start(mode === 'auto-train' ? { type: mode, input: inputs.split('\n').map(value => value.trim()).filter(Boolean), name: name.trim(), mode: 'auto', tokenizer: 'native-bpe', seq_len: 512 } : { type: mode, config: config.trim() }) }
    catch { /* useJob owns the visible error and always releases pending state. */ }
  }
  const canStart = mode === 'auto-train' ? !!inputs.trim() && !!name.trim() : !!config.trim()
  const knownJobs = (jobs.data?.jobs ?? []).filter((job: any) => ['train', 'pretrain', 'pipeline', 'auto-train'].includes(job.type))
  return <div style={{ padding: 28 }}><PageHeader title="Train" subtitle={monitor.jobId ? 'Job ' + monitor.jobId : 'Choose a configuration and start a run.'}
    right={<span className="chip">{monitor.job?.done ? monitor.job.code === 0 ? 'Completed' : 'Exit ' + monitor.job.code : monitor.running ? 'Running / checking status' : 'Idle'}</span>} />
    <div className="studio-split" style={{ display: 'grid', gridTemplateColumns: '300px minmax(300px,1fr)', gap: 14 }}>
      <Card><CardTitle>Configuration</CardTitle><Field label="Mode"><select className="select" value={mode} onChange={event => setMode(event.target.value)}>{['train', 'pretrain', 'pipeline', 'auto-train'].map(value => <option key={value}>{value}</option>)}</select></Field>
        {mode === 'auto-train' ? <><Field label="Input paths (one per line)"><textarea className="textarea" rows={4} value={inputs} onChange={event => setInputs(event.target.value)} /></Field><Field label="Run name"><input className="input" value={name} onChange={event => setName(event.target.value)} /></Field></> :
          <><Field label="Config file"><input className="input" value={config} onChange={event => setConfig(event.target.value)} /></Field><Field label="Template"><select className="select" defaultValue="" onChange={event => event.target.value && setConfig(event.target.value)}><option value="">Pick template…</option>{(templates.data?.templates ?? []).map((value: string) => <option key={value}>{value}</option>)}</select></Field></>}
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}><Button variant="primary" icon={<Play size={13} />} onClick={start} disabled={monitor.running || !canStart}>Start</Button><Button icon={<Square size={13} />} onClick={monitor.stop} disabled={!monitor.jobId || monitor.job?.done || monitor.pending}>Stop</Button></div>
        <StatusText msg={monitor.error ? 'Error: ' + monitor.error : ''} />
        {monitor.error && monitor.jobId && <Button onClick={() => monitor.select(null)}>Disconnect monitor</Button>}
        {knownJobs.length > 0 && <Field label="Reconnect to job"><select className="select" value={monitor.jobId ?? ''} onChange={event => monitor.select(event.target.value || null)}><option value="">Select job…</option>{knownJobs.map((job: any) => <option key={job.job_id} value={job.job_id}>{job.type} · {job.job_id} · {job.done ? 'finished' : 'running'}</option>)}</select></Field>}
        {jobs.error && <StatusText msg={'Error listing jobs: ' + errorMessage(jobs.error)} />}
        {monitor.job?.done && <Button onClick={() => monitor.select(null)}>New run</Button>}
      </Card>
      <div><Card><CardTitle>Metrics {metrics.data?.run ? '· ' + metrics.data.run : ''}</CardTitle>
        {metrics.error && <div role="alert">{errorMessage(metrics.error)}</div>}
        {rows.length > 0 ? <><div style={{ display: 'flex', gap: 18 }}><span>Step {latest.step}</span><span>Loss {fmt(latest.loss)}</span><span>Val {fmt(latest.val_loss)}</span></div><ResponsiveContainer width="100%" height={150}><LineChart data={rows.slice(-300)}><XAxis dataKey="step" /><YAxis /><Tooltip /><Line dataKey="loss" stroke="var(--color-neg)" dot={false} /><Line dataKey="val_loss" stroke="var(--color-warn)" dot={false} /></LineChart></ResponsiveContainer></> : <p>No metrics for this job yet.</p>}
      </Card><Card><CardTitle right={<Button icon={<RefreshCw size={12} />} onClick={() => { setShowLogs(true); monitor.refetch() }}>Refresh</Button>}>Output log</CardTitle><Button onClick={() => setShowLogs(value => !value)}>{showLogs ? 'Hide log' : 'Show log'}</Button>{showLogs && <LogBox lines={(monitor.job?.lines ?? []).slice(-600)} running={monitor.running} />}</Card></div>
    </div>
  </div>
}
