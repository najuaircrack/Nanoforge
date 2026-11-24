import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Operator from './pages/Operator'
import Evaluations from './pages/Evaluations'
import Artifacts from './pages/Artifacts'
const response = (data: any) => new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } })
function mount(element: React.ReactNode) { return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>{element}</QueryClientProvider>) }
beforeEach(() => { sessionStorage.clear(); localStorage.clear() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })
const project = { schema_version: 2, name: 'synthetic-project', workspace_root: '.', provider_kind: 'manual', enabled_tools: ['repo.apply_patch'], capabilities: ['workspace.write'], approval_mode: 'interactive', budgets: { max_steps: 3 }, resources: { context_paths: ['synthetic.txt'] } }

describe('operator actions, approvals and persisted trace', () => {
  it('loads bounded context, sends canonical actions and waits for an explicit approval before replay', async () => {
    const calls: { url: string; body?: any }[] = []
    let approved = false
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      const body = init?.body ? JSON.parse(init.body) : undefined; calls.push({ url, body })
      if (url === '/api/projects') return response({ projects: [{ path: 'synthetic.project.json', name: project.name, provider_kind: 'manual' }] })
      if (url === '/api/projects/load') return response({ project, tools: [{ name: 'repo.apply_patch' }] })
      if (url.startsWith('/api/projects/context')) return response({ files: [{ path: 'synthetic.txt', sha256: 'synthetic-hash', text: 'SYNTHETIC_CONTEXT', truncated: false }] })
      if (url === '/api/agent/runs') return response({ run_id: 'synthetic-run' })
      if (url === '/api/agent/runs/synthetic-run/approve') { approved = body.approved; return response({ ok: true }) }
      if (url === '/api/agent/runs/synthetic-run') return response({ run: { run_id: 'synthetic-run', project_path: 'synthetic.project.json', state: approved ? 'completed' : 'waiting_approval', done: approved, pending_approvals: approved ? [] : [{ approval_id: 'synthetic-approval', tool: 'repo.apply_patch', arguments: { path: 'synthetic.txt', content: 'CHANGE' } }], result: approved ? { answer: 'SYNTHETIC_RESULT' } : null, resources: { steps_used: 1 } } })
      if (url === '/api/agent/replay') return response({ replay: { status: 'completed', answer: 'REPLAY_RESULT', actions: [] } })
      if (url.endsWith('/trace')) return response({ events: [{ sequence: 0, stage: 'request', status: 'accepted', event: 'run', data: { redacted: true } }] })
      return response({ ok: true })
    }))
    mount(<Operator />)
    fireEvent.change(screen.getByRole('textbox', { name: 'Project manifest path' }), { target: { value: 'synthetic.project.json' } })
    fireEvent.click(screen.getByRole('button', { name: 'Load project' }))
    expect(await screen.findByText('synthetic-project', { selector: 'strong' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Preview context' }))
    expect(await screen.findByText(/SYNTHETIC_CONTEXT/)).toBeTruthy()
    fireEvent.change(screen.getByRole('textbox', { name: 'Agent request' }), { target: { value: 'Change the synthetic fixture' } })
    const actions = [{ schema_version: 1, type: 'tool_call', name: 'repo.apply_patch', arguments: { path: 'synthetic.txt', content: 'CHANGE' } }, { schema_version: 1, type: 'final', content: 'done' }]
    fireEvent.change(screen.getByRole('textbox', { name: 'Manual action sequence (JSON)' }), { target: { value: JSON.stringify(actions) } })
    fireEvent.click(screen.getByRole('button', { name: 'Start agent' }))
    const approval = await screen.findByRole('button', { name: 'Approve action' })
    expect(calls.find(call => call.url === '/api/agent/runs')?.body.actions).toEqual(actions)
    expect(calls.some(call => call.url.endsWith('/approve'))).toBe(false)
    fireEvent.click(approval)
    expect(await screen.findByText('SYNTHETIC_RESULT')).toBeTruthy()
    expect(calls.find(call => call.url.endsWith('/approve'))?.body).toEqual({ approval_id: 'synthetic-approval', approved: true })
    fireEvent.click(screen.getByRole('button', { name: 'Replay trace' }))
    expect(await screen.findByText(/REPLAY_RESULT/)).toBeTruthy()
    expect(calls.find(call => call.url === '/api/agent/replay')?.body).toEqual({ run_id: 'synthetic-run' })
  })
  it('loads server defaults, saves them through validation, and does not start an unsaved project', async () => {
    const saved: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      if (url === '/api/projects/schema') return response({ defaults: project })
      if (url === '/api/projects/save') { saved.push(JSON.parse(init.body)); return response({ path: 'synthetic.project.json', project, tools: [] }) }
      return response({ projects: [] })
    }))
    mount(<Operator />)
    fireEvent.click(screen.getByRole('button', { name: 'New project' }))
    expect(await screen.findByText(project.name, { selector: 'strong' })).toBeTruthy()
    fireEvent.change(screen.getByRole('textbox', { name: 'Agent request' }), { target: { value: 'synthetic request' } })
    expect((screen.getByRole('button', { name: 'Start agent' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.change(screen.getByRole('textbox', { name: 'Project manifest path' }), { target: { value: 'synthetic.project.json' } })
    fireEvent.click(screen.getByText('Edit manifest'))
    fireEvent.click(screen.getByRole('button', { name: 'Save project' }))
    await waitFor(() => expect(saved[0]).toEqual({ path: 'synthetic.project.json', project }))
  })
})

describe('evaluation and artifact service contracts', () => {
  it('starts a real suite job and loads failures and comparisons from the API', async () => {
    const submitted: any[] = [], compared: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      if (url === '/api/evaluations/suites') return response({ suites: [{ path: 'synthetic-suite.jsonl', name: 'Synthetic suite' }] })
      if (url === '/api/model/list') return response({ models: [{ path: 'synthetic.pt' }] })
      if (url === '/api/model/imports') return response({ models: [] })
      if (url === '/api/evaluations/run') { submitted.push(JSON.parse(init.body)); return response({ job_id: 'synthetic-eval' }) }
      if (url === '/api/jobs/synthetic-eval') return response({ done: true, code: 0, lines: ['Evaluation complete'] })
      if (url === '/api/evaluations/reports') return response({ reports: [{ path: 'report-a.json', summary: { pass_rate: 0.5 } }, { path: 'report-b.json', summary: { pass_rate: 1 } }] })
      if (url.startsWith('/api/evaluations/report?')) return response({ report: { summary: { total: 1, passed: 0, failed: 1, errors: 0, pass_rate: 0 }, cases: [{ case_id: 'synthetic-case', status: 'failed', latency_ms: 2, error: 'SYNTHETIC_FAILURE' }] } })
      if (url === '/api/evaluations/compare') { compared.push(JSON.parse(init.body)); return response({ comparison: { score_delta: 0.5 } }) }
      return response({ ok: true })
    }))
    mount(<Evaluations />)
    fireEvent.change(screen.getByRole('textbox', { name: 'Task suite path' }), { target: { value: 'synthetic-suite.jsonl' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Model reference' }), { target: { value: 'synthetic.pt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Run evaluation' }))
    await waitFor(() => expect(submitted[0]).toMatchObject({ suite_path: 'synthetic-suite.jsonl', model_ref: 'synthetic.pt' }))
    fireEvent.click((await screen.findAllByRole('button', { name: 'Open report' }))[0])
    expect(await screen.findByText('synthetic-case')).toBeTruthy()
    fireEvent.click(screen.getByText('Inspect case'))
    expect(screen.getByText(/SYNTHETIC_FAILURE/)).toBeTruthy()
    fireEvent.click(screen.getByLabelText('Compare report-a.json'))
    fireEvent.click(screen.getByLabelText('Compare report-b.json'))
    fireEvent.click(screen.getByRole('button', { name: 'Compare selected reports' }))
    await waitFor(() => expect(compared[0]).toEqual({ report_paths: ['report-a.json', 'report-b.json'] }))
    expect(await screen.findByText(/score_delta/)).toBeTruthy()
  })
  it('shows recorded artifact provenance without inventing missing governance', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response({ artifacts: [{ path: 'synthetic.manifest.json', manifest: { schema_version: 1, kind: 'checkpoint', artifact_id: 'synthetic-id', content_fingerprint: 'synthetic-fingerprint', parents: ['synthetic-parent'] } }] })))
    mount(<Artifacts />)
    fireEvent.click(await screen.findByRole('button', { name: /synthetic.manifest.json/ }))
    expect(screen.getAllByText('Not recorded').length).toBe(2)
    expect(screen.getAllByText(/synthetic-fingerprint/).length).toBeGreaterThan(0)
    expect(screen.getByText(/synthetic-parent/)).toBeTruthy()
  })
})
