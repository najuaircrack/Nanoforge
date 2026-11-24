import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Chat from './pages/Chat'
import Data from './pages/Data'
import Configs from './pages/Configs'
import ConfigBuilder from './pages/ConfigBuilder'
import Train from './pages/Train'
import { useJob } from './lib/useJob'

const response = (data: any, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } })
function mount(element: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  return render(<QueryClientProvider client={client}>{element}</QueryClientProvider>)
}
beforeEach(() => { localStorage.clear(); sessionStorage.clear(); Element.prototype.scrollIntoView = vi.fn() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('chat recovery and errors', () => {
  it('allows reloading the same remembered model after an empty server status and selecting imports', async () => {
    localStorage.setItem('nanoforge_loaded_model', 'runs/synthetic.pt')
    const requests: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      requests.push({ url, init })
      if (url === '/api/chat/status') return response({ ok: true, loaded: null })
      if (url === '/api/model/list') return response({ ok: true, models: [{ path: 'runs/synthetic.pt' }] })
      if (url === '/api/model/imports') return response({ ok: true, models: [{ name: 'synthetic-import' }] })
      if (url === '/api/chat/load') return response({ ok: true })
      return response({ ok: true, loaded: null })
    }))
    mount(<Chat />)
    await waitFor(() => expect((screen.getByRole('button', { name: 'Select model' }) as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(screen.getByRole('button', { name: 'Select model' }))
    fireEvent.click(await screen.findByRole('button', { name: 'runs/synthetic.pt' }))
    await waitFor(() => expect(requests.some(r => r.url === '/api/chat/load' && JSON.parse(r.init.body).path === 'runs/synthetic.pt')).toBe(true))
    await waitFor(() => expect((screen.getByRole('button', { name: 'Select model' }) as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(screen.getByRole('button', { name: 'Select model' }))
    fireEvent.click(await screen.findByRole('button', { name: 'synthetic-import' }))
    await waitFor(() => expect(requests.some(r => r.url === '/api/chat/load' && JSON.parse(r.init.body).path === 'synthetic-import')).toBe(true))
  })
  it('releases loading controls and displays a model-load HTTP failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/model/list') return response({ models: [{ path: 'synthetic.pt' }] })
      if (url === '/api/model/imports') return response({ models: [] })
      if (url === '/api/chat/load') return response({ detail: 'load refused' }, 500)
      return response({ ok: true, loaded: null })
    }))
    mount(<Chat />)
    await waitFor(() => expect((screen.getByRole('button', { name: 'Select model' }) as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(screen.getByRole('button', { name: 'Select model' }))
    fireEvent.click(await screen.findByRole('button', { name: 'synthetic.pt' }))
    expect(await screen.findByText('Error: load refused')).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Select model' }) as HTMLButtonElement).disabled).toBe(false)
  })
  it('turns a non-SSE chat HTTP error into a visible message and permits retry', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/chat/status') return response({ loaded: 'synthetic.pt' })
      if (url === '/api/chat/stream') return response({ detail: 'generation failed' }, 500)
      return response({ models: [], loaded: null })
    }))
    mount(<Chat />)
    await waitFor(() => expect((screen.getByRole('textbox', { name: 'Message' }) as HTMLTextAreaElement).disabled).toBe(false))
    fireEvent.change(screen.getByRole('textbox', { name: 'Message' }), { target: { value: 'synthetic prompt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(screen.getAllByText('Error: generation failed').length).toBeGreaterThan(0))
    expect((screen.getByRole('textbox', { name: 'Message' }) as HTMLTextAreaElement).disabled).toBe(false)
  })
})

describe('configuration and data failure paths', () => {
  it('preserves an existing config draft when raw loading returns HTTP 400', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/config/templates') return response({ templates: ['configs/missing.yaml'] })
      return response({ detail: 'missing synthetic file' }, 400)
    }))
    mount(<Configs />)
    fireEvent.click(screen.getByRole('button', { name: 'New config' }))
    const editor = screen.getByRole('textbox', { name: 'YAML configuration' }) as HTMLTextAreaElement
    const before = editor.value
    fireEvent.click(await screen.findByRole('button', { name: 'configs/missing.yaml' }))
    expect(await screen.findByText('Error: missing synthetic file')).toBeTruthy()
    expect(editor.value).toBe(before)
  })
  it('retains each upload failure when another upload succeeds', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      if (url === '/api/data/upload') return init.body.get('file').name === 'bad.txt' ? response({ ok: false, error: 'synthetic rejection' }) : response({ ok: true, path: 'data/uploads/good.txt' })
      return response({ files: [], catalog: [] })
    }))
    mount(<Data />)
    await userEvent.upload(screen.getByLabelText('Upload files'), [new File(['a'], 'good.txt'), new File(['b'], 'bad.txt')])
    expect(await screen.findByText('1 uploaded, 1 failed')).toBeTruthy()
    expect(screen.getByText('bad.txt: synthetic rejection')).toBeTruthy()
    expect((screen.getByLabelText('Upload files') as HTMLInputElement).value).toBe('')
  })
  it('keeps the selected file profile when an older request finishes later', async () => {
    let resolveA: ((value: Response) => void) | undefined
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      if (url.startsWith('/api/data/list')) return response({ files: [{ path: 'A', size: 1 }, { path: 'B', size: 1 }] })
      if (url === '/api/data/profile') {
        const path = JSON.parse(init.body).path
        if (path === 'A') return new Promise<Response>(resolve => { resolveA = resolve })
        return response({ profile: { source: 'PROFILE_B' } })
      }
      return response({ catalog: [] })
    }))
    mount(<Data />)
    fireEvent.click(await screen.findByRole('button', { name: 'Profile A' }))
    await waitFor(() => expect(resolveA).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Profile B' }))
    expect(await screen.findByText(/PROFILE_B/)).toBeTruthy()
    resolveA!(response({ profile: { source: 'PROFILE_A' } }))
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(screen.queryByText(/PROFILE_A/)).toBeNull()
    expect(screen.getByText(/PROFILE_B/)).toBeTruthy()
  })
  it('never commits an older live-build response over newer values', async () => {
    const requests: { body: any; resolve: (value: Response) => void }[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      if (url === '/api/config/schema') return response({ schema: { model: { label: 'Model', fields: [{ name: 'd_model', type: 'int', widget: 'number', default: 32 }] } } })
      if (url === '/api/config/templates') return response({ templates: [] })
      return new Promise<Response>(resolve => { requests.push({ body: JSON.parse(init.body), resolve }) })
    }))
    mount(<ConfigBuilder />)
    const input = await screen.findByRole('spinbutton', { name: 'D Model' })
    await waitFor(() => expect(requests.length).toBe(1))
    fireEvent.change(input, { target: { value: '64' } })
    await waitFor(() => expect(requests.length).toBe(2))
    requests[1].resolve(response({ ok: true, yaml: 'CURRENT_64' }))
    expect(await screen.findByText('CURRENT_64')).toBeTruthy()
    requests[0].resolve(response({ ok: true, yaml: 'STALE_32' }))
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(screen.queryByText('STALE_32')).toBeNull()
    expect(screen.getByText('CURRENT_64')).toBeTruthy()
  })
})

describe('persistent job control', () => {
  function JobHarness() {
    const job = useJob('test')
    return <><button onClick={() => job.start({ type: 'train' }).catch(() => {})} disabled={job.running}>Start</button><button onClick={job.stop} disabled={!job.jobId || job.job?.done}>Stop</button><div>{job.error}</div><div>{job.jobId}</div></>
  }
  it('releases a failed job submission and restores active identity after navigation', async () => {
    let fail = true
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/jobs/start') return fail ? response({ detail: 'start failed' }, 500) : response({ job_id: 'synthetic-job', run_dir: 'runs/synthetic' })
      return response({ done: false, code: null, lines: [] })
    }))
    const view = mount(<JobHarness />)
    fireEvent.click(screen.getByRole('button', { name: 'Start' }))
    expect(await screen.findByText('start failed')).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Start' }) as HTMLButtonElement).disabled).toBe(false)
    fail = false; fireEvent.click(screen.getByRole('button', { name: 'Start' }))
    expect(await screen.findByText('synthetic-job')).toBeTruthy()
    view.unmount(); mount(<JobHarness />)
    expect(screen.getByText('synthetic-job')).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Start' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement).disabled).toBe(false)
  })
  it('requests metrics for the persisted job and submits auto-train using its real input schema', async () => {
    sessionStorage.setItem('latticekin_job_training', 'synthetic-old')
    const urls: string[] = [], submissions: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      urls.push(url)
      if (url === '/api/jobs/start') { submissions.push(JSON.parse(init.body)); return response({ job_id: 'synthetic-new' }) }
      if (url === '/api/jobs/synthetic-old') return response({ done: true, code: 0, lines: [] })
      if (url === '/api/jobs') return response({ jobs: [] })
      return response({ rows: [], templates: [] })
    }))
    mount(<Train />)
    await waitFor(() => expect(urls).toContain('/api/metrics?job_id=synthetic-old'))
    fireEvent.click(await screen.findByRole('button', { name: 'New run' }))
    fireEvent.change(screen.getByRole('combobox', { name: 'Mode' }), { target: { value: 'auto-train' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Input paths (one per line)' }), { target: { value: 'synthetic.jsonl' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Run name' }), { target: { value: 'synthetic' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start' }))
    await waitFor(() => expect(submissions[0]).toMatchObject({ type: 'auto-train', input: ['synthetic.jsonl'], name: 'synthetic' }))
    expect(submissions[0].config).toBeUndefined()
  })
})
