import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { get, post, errorMessage } from '../lib/api'
import { Card, CardTitle, PageHeader, Field, Button, StatusText, LogBox, Empty } from '../lib/ui'
import { useJob } from '../lib/useJob'
import { fmt, fmtPct } from '../lib/metrics'
export default function Evaluations() {
  const [suite, setSuite] = useState('configs/evals/smoke.jsonl'), [model, setModel] = useState(''), [output, setOutput] = useState('reports/evaluations')
  const [reportPath, setReportPath] = useState(''), [compared, setCompared] = useState<string[]>([]), [comparison, setComparison] = useState<any>(null)
  const [status, setStatus] = useState(''), [pending, setPending] = useState(false)
  const monitor = useJob('evaluation')
  const suites = useQuery({ queryKey: ['eval-suites'], queryFn: () => get('/api/evaluations/suites') })
  const models = useQuery({ queryKey: ['models'], queryFn: () => get('/api/model/list') })
  const imports = useQuery({ queryKey: ['imports'], queryFn: () => get('/api/model/imports') })
  const reports = useQuery({ queryKey: ['evaluation-reports'], queryFn: () => get('/api/evaluations/reports') })
  const report = useQuery({ queryKey: ['evaluation-report', reportPath], enabled: !!reportPath, queryFn: ({ signal }) => get('/api/evaluations/report', { path: reportPath }, { signal }) })
  const entries = [...(models.data?.models ?? []), ...(imports.data?.models ?? []).map((item: any) => ({ ...item, ref: item.ref ?? item.name }))]
  const refs: string[] = Array.from(new Set(entries.map((item: any) => item.ref ?? item.path ?? item.name)))
  useEffect(() => { if (monitor.job?.done) reports.refetch() }, [monitor.job?.done])
  async function start() {
    setPending(true); setStatus('')
    try { const result = await post('/api/evaluations/run', { suite_path: suite.trim(), model_ref: model.trim(), output_dir: output.trim() }); if (typeof result.job_id !== 'string' || !result.job_id) throw new Error('Server returned no job identity'); monitor.select(result.job_id); setStatus('Evaluation started') }
    catch (error) { setStatus('Error: ' + errorMessage(error)) } finally { setPending(false) }
  }
  async function compare() {
    setPending(true); setStatus('')
    try { const result = await post('/api/evaluations/compare', { report_paths: compared }); setComparison(result.comparison); setStatus('Reports compared') }
    catch (error) { setStatus('Error: ' + errorMessage(error)) } finally { setPending(false) }
  }
  const loaded = report.data?.report, cases = report.data?.cases ?? loaded?.cases ?? [], summary = loaded?.summary ?? {}
  return <div style={{ padding: 28 }}><PageHeader title="Evaluations" subtitle="Repeatable task results, failures, resource observations and comparable reports." /><StatusText msg={status} />
    <div className="studio-split" style={{ display: 'grid', gridTemplateColumns: '320px minmax(300px,1fr)', gap: 14 }}>
      <Card><CardTitle>Run a task suite</CardTitle><Field label="Task suite path"><input className="input" value={suite} onChange={event => setSuite(event.target.value)} /></Field>
        <Field label="Available suites"><select className="select" defaultValue="" onChange={event => setSuite(event.target.value)}><option value="">Select suite…</option>{(suites.data?.suites ?? []).map((item: any) => <option key={item.path} value={item.path}>{item.name ?? item.path}</option>)}</select></Field>
        <Field label="Model reference"><input className="input" value={model} onChange={event => setModel(event.target.value)} /></Field>
        <Field label="Available models"><select className="select" value={model} onChange={event => setModel(event.target.value)}><option value="">Select model…</option>{refs.map(ref => <option key={ref}>{ref}</option>)}</select></Field>
        <Field label="Report output directory"><input className="input" value={output} onChange={event => setOutput(event.target.value)} /></Field>
        <Button onClick={start} variant="primary" disabled={pending || monitor.running || !suite.trim() || !model.trim() || !output.trim()}>Run evaluation</Button>
        {monitor.jobId && !monitor.job?.done && <Button onClick={monitor.stop} disabled={monitor.pending}>Stop evaluation</Button>}
        {suites.error && <StatusText msg={'Error: ' + errorMessage(suites.error)} />}
        <StatusText msg={monitor.error ? 'Error: ' + monitor.error : monitor.job?.done ? monitor.job.code === 0 ? '✓ Evaluation job completed' : 'Error: evaluation exit ' + monitor.job.code : ''} />
        {monitor.error && monitor.jobId && <Button onClick={() => monitor.select(null)}>Disconnect monitor</Button>}
        {monitor.jobId && <LogBox lines={(monitor.job?.lines ?? []).slice(-600)} running={monitor.running} minHeight={140} />}
      </Card>
      <div><Card><CardTitle right={<Button onClick={() => reports.refetch()}>Refresh reports</Button>}>Saved reports</CardTitle>
        {reports.error && <StatusText msg={'Error: ' + errorMessage(reports.error)} />}
        {!reports.data?.reports?.length && <Empty>No evaluation reports yet.</Empty>}
        {(reports.data?.reports ?? []).map((item: any) => <div key={item.path} style={{ display: 'flex', gap: 10, padding: '8px 0', alignItems: 'center' }}>
          <label><input type="checkbox" checked={compared.includes(item.path)} onChange={event => { setCompared(current => event.target.checked ? [...current, item.path] : current.filter(path => path !== item.path)); setComparison(null) }} /> Compare {item.path}</label>
          <Button onClick={() => setReportPath(item.path)}>Open report</Button>
          {item.summary?.pass_rate != null && <span>{fmtPct(item.summary.pass_rate)}</span>}
        </div>)}
        <Button onClick={compare} disabled={pending || compared.length < 2}>Compare selected reports</Button>
        {comparison && <div style={{ marginTop: 14 }}>
          {comparison.compatible === false && <p role="alert">Reports use incompatible suites. Scores cannot be compared.</p>}
          {(comparison.candidates ?? []).length > 0 && <table style={{ width: '100%', fontSize: 12 }}><thead><tr><th align="left">Candidate</th><th align="left">Pass rate change</th><th align="left">Latency change</th><th align="left">Regressions</th><th align="left">Thresholds</th></tr></thead><tbody>{comparison.candidates.map((candidate: any) => <tr key={candidate.report_id}><td>{candidate.report_id}</td><td>{fmtPct(candidate.pass_rate_delta)}</td><td>{fmt(candidate.mean_latency_delta_ms, 1)} ms</td><td>{candidate.regressions?.length ?? 0}</td><td>{candidate.thresholds_passed === true ? 'Passed' : candidate.thresholds_passed === false ? 'Failed' : 'Not recorded'}</td></tr>)}</tbody></table>}
          <details open={!(comparison.candidates ?? []).length}><summary>Comparison evidence and thresholds</summary><pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(comparison, null, 2)}</pre></details>
        </div>}
      </Card>
      {reportPath && <Card><CardTitle>Report {reportPath}</CardTitle>{report.error && <StatusText msg={'Error: ' + errorMessage(report.error)} />}{loaded && <>
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}><span>Cases {summary.total}</span><span>Passed {summary.passed}</span><span>Failed {summary.failed}</span><span>Errors {summary.errors}</span><span>Pass rate {fmtPct(summary.pass_rate)}</span></div>
        <details><summary>Report provenance and metrics</summary><pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify({ ...loaded, cases: undefined }, null, 2)}</pre></details>
        <table style={{ width: '100%', fontSize: 12, marginTop: 14 }}><thead><tr><th align="left">Case</th><th align="left">Result</th><th align="left">Latency</th><th align="left">Evidence</th></tr></thead><tbody>{cases.map((item: any, index: number) => <tr key={item.case_id ?? item.id ?? index}><td>{item.case_id ?? item.id}</td><td style={{ color: item.status === 'passed' ? 'var(--color-pos)' : 'var(--color-neg)' }}>{item.status}</td><td>{fmt(item.latency_ms, 1)} ms</td><td><details><summary>Inspect case</summary><pre style={{ whiteSpace: 'pre-wrap', maxWidth: 480, fontSize: 12 }}>{JSON.stringify(item, null, 2)}</pre></details></td></tr>)}</tbody></table>
      </>}</Card>}
      </div>
    </div>
  </div>
}
