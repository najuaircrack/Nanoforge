import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Zap, Download, RefreshCw } from 'lucide-react'
import { get, post, fmtBytes } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Field, LogBox, Empty } from '../lib/ui'

const DATASETS = [
  { id: 'openwebtext',  name: 'OpenWebText',  size: '~40 GB',  desc: 'Web text corpus' },
  { id: 'tinystories',  name: 'TinyStories',  size: '~2 GB',   desc: 'Simple children stories' },
  { id: 'the-pile',     name: 'The Pile',     size: '~800 GB', desc: 'Diverse text corpus' },
  { id: 'code-parrot',  name: 'CodeParrot',   size: '~50 GB',  desc: 'GitHub code' },
  { id: 'alpaca',       name: 'Alpaca',       size: '~50 MB',  desc: 'Instruction-following data' },
  { id: 'dolly',        name: 'Dolly 15k',    size: '~13 MB',  desc: 'Instruction dataset' },
]

type Tab = 'browse' | 'download' | 'process'

export default function Data() {
  const [tab, setTab] = useState<Tab>('browse')
  const [root, setRoot] = useState('data')
  const [selected, setSelected] = useState<string | null>(null)
  const [profile, setProfile] = useState<any>(null)
  const [profiling, setProfiling] = useState(false)
  const [jobLog, setJobLog] = useState<string[]>([])
  const [jobRunning, setJobRunning] = useState(false)
  const [procInput, setProcInput] = useState('')
  const [procOut, setProcOut] = useState('data/packed/my-dataset')
  const [procMode, setProcMode] = useState('auto')
  const [procTokenizer, setProcTokenizer] = useState('byte')
  const [procSeqLen, setProcSeqLen] = useState(512)

  const { data: fileData, refetch } = useQuery({
    queryKey: ['data-list', root],
    queryFn: () => get('/api/data/list', { root }),
  })
  const files: any[] = fileData?.files ?? []

  async function profileFile(path: string) {
    setProfiling(true); setProfile(null)
    const r = await post('/api/data/profile', { path })
    setProfiling(false)
    if (r.ok) setProfile(r.profile)
  }

  function startSSE(jobId: string) {
    const es = new EventSource(`/stream/job/${jobId}`)
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'line') setJobLog(l => [...l, msg.line])
      else if (msg.type === 'done') { setJobRunning(false); es.close() }
    }
    es.onerror = () => { setJobRunning(false); es.close() }
  }

  async function runProcess() {
    if (!procInput) return
    setJobRunning(true); setJobLog([])
    const r = await post('/api/jobs/start', {
      type: 'prepare',
      input: procInput.split('\n').filter(Boolean),
      out: procOut, mode: procMode, tokenizer: procTokenizer, seq_len: procSeqLen,
    })
    if (!r.ok) { setJobLog([`Error: ${r.error}`]); setJobRunning(false); return }
    startSSE(r.job_id)
  }

  async function downloadDataset(id: string) {
    setJobRunning(true); setJobLog([`Starting download: ${id}...`])
    const r = await post('/api/jobs/start', { type: 'download', dataset: id })
    if (!r.ok) { setJobLog([`Error: ${r.error}`]); setJobRunning(false); return }
    startSSE(r.job_id)
  }

  const TABS: { id: Tab; label: string }[] = [
    { id: 'browse', label: 'Browse' },
    { id: 'download', label: 'Download' },
    { id: 'process', label: 'Process' },
  ]

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Data"
        right={
          <div style={{ display: 'flex', gap: 2, background: 'var(--color-surface-2)', padding: 3, borderRadius: 10, border: '1px solid var(--color-line)' }}>
            {TABS.map(t => (
              <button key={t.id} onClick={() => setTab(t.id)} style={{
                padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
                fontSize: 13, fontWeight: 500,
                background: tab === t.id ? 'var(--color-accent)' : 'transparent',
                color: tab === t.id ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
              }}>{t.label}</button>
            ))}
          </div>
        }
      />

      {tab === 'browse' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 14 }}>
          <Card>
            <CardTitle right={
              <Button variant="ghost" onClick={() => refetch()} icon={<RefreshCw size={12} />} style={{ fontSize: 12, padding: '3px 8px' }}>
                Refresh
              </Button>
            }>
              Files in "{root}"
            </CardTitle>
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              <input value={root} onChange={e => setRoot(e.target.value)} className="input" placeholder="data root" style={{ flex: 1 }} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 480, overflow: 'auto' }}>
              {files.length === 0 && <Empty>No files found in "{root}"</Empty>}
              {files.map(f => (
                <div key={f.path} onClick={() => { setSelected(f.path); profileFile(f.path) }}
                  style={{
                    padding: '7px 10px', borderRadius: 8, cursor: 'pointer',
                    background: selected === f.path ? 'var(--color-surface-3)' : 'transparent',
                    border: `1px solid ${selected === f.path ? 'var(--color-line-2)' : 'transparent'}`,
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--color-ink-2)' }}>{f.path}</span>
                  <span style={{ fontSize: 11, color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)' }}>{fmtBytes(f.size)}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <CardTitle>Profile</CardTitle>
            {profiling && <div style={{ color: 'var(--color-ink-2)', fontSize: 13 }}>Profiling...</div>}
            {!profiling && !profile && <Empty>Select a file to profile.</Empty>}
            {profile && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                {Object.entries(profile).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '5px 0', borderBottom: '1px solid var(--color-line)' }}>
                    <span style={{ color: 'var(--color-ink-2)' }}>{k}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ink)', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{String(v)}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {tab === 'download' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
            {DATASETS.map(d => (
              <Card key={d.id} pad={18}>
                <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 3 }}>{d.name}</div>
                <div style={{ fontSize: 12, color: 'var(--color-ink-2)', marginBottom: 2 }}>{d.desc}</div>
                <div style={{ fontSize: 11, color: 'var(--color-ink-3)', marginBottom: 14, fontFamily: 'var(--font-mono)' }}>{d.size}</div>
                <Button onClick={() => downloadDataset(d.id)} disabled={jobRunning} icon={<Download size={12} />}>
                  Download
                </Button>
              </Card>
            ))}
          </div>
          <LogBox lines={jobLog} running={jobRunning} />
        </div>
      )}

      {tab === 'process' && (
        <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 14 }}>
          <Card>
            <CardTitle>Prepare dataset</CardTitle>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <Field label="Input paths (one per line)">
                <textarea value={procInput} onChange={e => setProcInput(e.target.value)} rows={4} className="textarea" placeholder="data/raw/my-data.jsonl" />
              </Field>
              <Field label="Output directory">
                <input value={procOut} onChange={e => setProcOut(e.target.value)} className="input" />
              </Field>
              <Field label="Mode">
                <select value={procMode} onChange={e => setProcMode(e.target.value)} className="select">
                  {['auto','generative','chat','instruct','completion','code'].map(m => <option key={m}>{m}</option>)}
                </select>
              </Field>
              <Field label="Tokenizer">
                <select value={procTokenizer} onChange={e => setProcTokenizer(e.target.value)} className="select">
                  {['byte','bpe','native-bpe','wordpiece','sentencepiece'].map(t => <option key={t}>{t}</option>)}
                </select>
              </Field>
              <Field label={`Seq len: ${procSeqLen}`}>
                <input type="range" min={128} max={4096} step={128} value={procSeqLen} onChange={e => setProcSeqLen(+e.target.value)} style={{ width: '100%' }} />
              </Field>
              <Button onClick={runProcess} disabled={jobRunning || !procInput} variant="primary" icon={<Zap size={13} />}>
                {jobRunning ? 'Processing...' : 'Process'}
              </Button>
            </div>
          </Card>
          <LogBox lines={jobLog} running={jobRunning} />
        </div>
      )}
    </div>
  )
}
