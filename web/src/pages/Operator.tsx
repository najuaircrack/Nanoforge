import { useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { get, post, errorMessage } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Field, StatusText, Empty } from '../lib/ui'
type Project = { name: string; provider_kind: string; model?: Record<string, any>; budgets?: Record<string, any>; resources?: { context_paths?: string[] }; enabled_tools?: string[]; capabilities?: string[]; approval_mode?: string }
export default function Operator() {
  const [path, setPath] = useState('configs/project-example.yaml'), [savedPath, setSavedPath] = useState(''), [project, setProject] = useState<Project | null>(null)
  const [projectDraft, setProjectDraft] = useState(''), [tools, setTools] = useState<any[]>([]), [request, setRequest] = useState('')
  const [actions, setActions] = useState('[]'), [runId, setRunId] = useState(() => sessionStorage.getItem('latticekin_agent_run') ?? '')
  const [status, setStatus] = useState(''), [pending, setPending] = useState(false), [replay, setReplay] = useState<any>(null), [context, setContext] = useState<any>(null)
  const loadRevision = useRef(0), client = useQueryClient()
  const projects = useQuery({ queryKey: ['projects'], queryFn: () => get('/api/projects') })
  const runQuery = useQuery({ queryKey: ['agent-run', runId], enabled: !!runId, queryFn: ({ signal }) => get('/api/agent/runs/' + encodeURIComponent(runId), undefined, { signal }), refetchInterval: query => query.state.data?.run?.done ? false : 1000 })
  const trace = useQuery({ queryKey: ['agent-trace', runId], enabled: !!runId, queryFn: ({ signal }) => get('/api/agent/runs/' + encodeURIComponent(runId) + '/trace', undefined, { signal }), refetchInterval: runQuery.data?.run?.done ? false : 1500 })
  const run = runQuery.data?.run
  async function perform(task: () => Promise<void>) { setPending(true); setStatus(''); try { await task() } catch (error) { setStatus('Error: ' + errorMessage(error)) } finally { setPending(false) } }
  async function load() {
    const revision = ++loadRevision.current
    await perform(async () => { const result = await post('/api/projects/load', { path: path.trim() }); if (revision !== loadRevision.current) return; setSavedPath(path.trim()); setProject(result.project); setProjectDraft(JSON.stringify(result.project, null, 2)); setTools(result.tools ?? []); setContext(null); setStatus('Loaded ' + result.project.name) })
  }
  async function create() {
    await perform(async () => { const result = await get('/api/projects/schema'); const value = result.default ?? result.project ?? result.defaults; if (!value) throw new Error('Server did not return a project template'); setSavedPath(''); setProject(value); setProjectDraft(JSON.stringify(value, null, 2)); setTools([]); setContext(null); setStatus('Edit the project settings and save a manifest.') })
  }
  async function save() {
    await perform(async () => { const value = JSON.parse(projectDraft); const result = await post('/api/projects/save', { path: path.trim(), project: value }); setSavedPath(path.trim()); setProject(result.project ?? value); setProjectDraft(JSON.stringify(result.project ?? value, null, 2)); setTools(result.tools ?? tools); setStatus('✓ Saved ' + result.path); await client.invalidateQueries({ queryKey: ['projects'] }) })
  }
  async function start() {
    await perform(async () => {
      let manual: any[] | undefined
      if (project?.provider_kind === 'manual') { manual = JSON.parse(actions); if (!Array.isArray(manual) || !manual.length) throw new Error('Manual projects require a non-empty action array') }
      const result = await post('/api/agent/runs', { project_path: path.trim(), request: request.trim(), ...(manual ? { actions: manual } : {}) })
      if (typeof result.run_id !== 'string' || !result.run_id) throw new Error('Server returned no run identity')
      setRunId(result.run_id); sessionStorage.setItem('latticekin_agent_run', result.run_id); setReplay(null); setStatus('Run started')
    })
  }
  async function decide(approvalId: string, approved: boolean) {
    await perform(async () => { await post('/api/agent/runs/' + encodeURIComponent(runId) + '/approve', { approval_id: approvalId, approved }); await runQuery.refetch(); setStatus(approved ? 'Action approved' : 'Action denied') })
  }
  function exportTrace() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(trace.data?.events ?? [], null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = 'agent-trace-' + runId + '.json'; link.click(); URL.revokeObjectURL(url)
  }
  return <div style={{ padding: 28 }}><PageHeader title="Agent operator" subtitle="Project context, structured actions, explicit approvals and recorded results." />
    <StatusText msg={status} />{projects.error && <div role="alert">{errorMessage(projects.error)}</div>}
    <div className="studio-split" style={{ display: 'grid', gridTemplateColumns: 'minmax(260px,360px) minmax(320px,1fr)', gap: 14 }}>
      <Card><CardTitle>Project</CardTitle><Field label="Project manifest path"><input className="input" disabled={pending} value={path} onChange={event => setPath(event.target.value)} /></Field>
        <Field label="Existing projects"><select className="select" defaultValue="" onChange={event => setPath(event.target.value)}><option value="">Choose a project…</option>{(projects.data?.projects ?? []).map((item: any) => <option key={item.path} value={item.path}>{item.name} · {item.provider_kind}</option>)}</select></Field>
        <div style={{ display: 'flex', gap: 8, margin: '12px 0' }}><Button onClick={load} disabled={pending || !path.trim()}>Load project</Button><Button onClick={create} disabled={pending}>New project</Button></div>
        {project && <><p><strong>{project.name}</strong> · {project.provider_kind}</p><p>Approval: {project.approval_mode}</p><small>Tools: {(project.enabled_tools ?? []).join(', ')}</small>
          <details style={{ marginTop: 14 }}><summary>Edit manifest</summary><Field label="Project settings JSON"><textarea className="textarea" rows={16} value={projectDraft} onChange={event => setProjectDraft(event.target.value)} /></Field><Button onClick={save} disabled={pending || !projectDraft || !path.trim()}>Save project</Button></details>
          <CardTitle>Budgets</CardTitle><pre style={{ fontSize: 12, whiteSpace: 'pre-wrap' }}>{JSON.stringify(project.budgets, null, 2)}</pre>
          <CardTitle>Selected context paths</CardTitle>{(project.resources?.context_paths ?? []).map(item => <div key={item}>{item}</div>)}
          <Button disabled={pending || savedPath !== path.trim()} onClick={() => perform(async () => { const result = await get('/api/projects/context', { path }); setContext(result.files); setStatus('Context inspected') })}>Preview context</Button>
          {context && <pre style={{ maxHeight: 300, overflow: 'auto', whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(context, null, 2)}</pre>}
          <details><summary>Tool schemas</summary><pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(tools, null, 2)}</pre></details>
        </>}
      </Card>
      <div><Card><CardTitle>Request</CardTitle><Field label="Agent request"><textarea className="textarea" rows={4} value={request} onChange={event => setRequest(event.target.value)} /></Field>
        {project?.provider_kind === 'manual' && <Field label="Manual action sequence (JSON)"><textarea className="textarea" rows={6} value={actions} onChange={event => setActions(event.target.value)} placeholder='[{"schema_version":1,"type":"tool_call","name":"repo.list_files","arguments":{}},{"schema_version":1,"type":"final","content":"Review complete"}]' /></Field>}
        <Button variant="primary" onClick={start} disabled={pending || !project || savedPath !== path.trim() || projectDraft !== JSON.stringify(project, null, 2) || !request.trim() || (!!runId && !run?.done)}>Start agent</Button>
        {runId && !run?.done && <Button onClick={() => perform(async () => { await post('/api/agent/runs/' + encodeURIComponent(runId) + '/cancel'); await runQuery.refetch(); setStatus('Cancellation requested') })} disabled={pending}>Cancel run</Button>}
      </Card>
      {runQuery.error && <Card><StatusText msg={'Error: ' + errorMessage(runQuery.error)} /><Button onClick={() => { setRunId(''); sessionStorage.removeItem('latticekin_agent_run') }}>Clear disconnected monitor</Button></Card>}
      {run && <Card><CardTitle>Run {run.run_id} · {run.state}</CardTitle><small>Project: {run.project_path}</small>{run.error && <StatusText msg={'Error: ' + run.error} />}
        {(run.pending_approvals ?? []).map((approval: any) => <div key={approval.approval_id} style={{ border: '1px solid var(--color-warn)', padding: 14, marginBottom: 12 }}><strong>Approval required: {approval.tool}</strong><pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(approval.arguments, null, 2)}</pre><div style={{ display: 'flex', gap: 8 }}><Button variant="primary" disabled={pending} onClick={() => decide(approval.approval_id, true)}>Approve action</Button><Button disabled={pending} onClick={() => decide(approval.approval_id, false)}>Deny action</Button></div></div>)}
        {run.result?.answer && <div style={{ whiteSpace: 'pre-wrap' }}>{run.result.answer}</div>}
        {run.result?.observations && <details><summary>Tool observations</summary><pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(run.result.observations, null, 2)}</pre></details>}
        <details open><summary>Resource use and remaining budgets</summary><pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(run.resources ?? run.result?.resources ?? {}, null, 2)}</pre></details>
        {run.done && <Button onClick={() => perform(async () => { const result = await post('/api/agent/replay', { run_id: runId }); setReplay(result.replay); setStatus('Replay reconstructed without running tools') })} disabled={pending}>Replay trace</Button>}
        {replay && <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(replay, null, 2)}</pre>}
      </Card>}
      {runId && <Card><CardTitle right={<Button onClick={exportTrace} disabled={!trace.data?.events?.length}>Export trace</Button>}>Recorded trace</CardTitle>{trace.error && <StatusText msg={'Error: ' + errorMessage(trace.error)} />}{!trace.data?.events?.length && <Empty>No events yet.</Empty>}{(trace.data?.events ?? []).slice(-1000).map((event: any, index: number) => <details key={event.sequence ?? index}><summary>{event.sequence ?? index} · {event.stage ?? event.phase ?? event.category} · {event.status} · {event.event ?? event.name}</summary><pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(event, null, 2)}</pre></details>)}</Card>}
      </div>
    </div>
  </div>
}
