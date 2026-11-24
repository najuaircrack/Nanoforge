import { useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Upload, Download, Zap } from 'lucide-react'
import { get, post, upload, errorMessage, fmtBytes } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Field, LogBox, Empty, Tabs, StatusText } from '../lib/ui'
import { useJob } from '../lib/useJob'
type Tab = 'browse' | 'download' | 'process'
type UploadResult = { name: string; path?: string; error?: string }
export default function Data() {
  const [tab, setTab] = useState<Tab>('browse'), [root, setRoot] = useState('data'), [selected, setSelected] = useState<string | null>(null)
  const [inputs, setInputs] = useState(''), [output, setOutput] = useState('data/packed/my-dataset'), [mode, setMode] = useState('auto'), [tokenizer, setTokenizer] = useState('native-bpe'), [seqLen, setSeqLen] = useState(512)
  const [uploading, setUploading] = useState(false), [results, setResults] = useState<UploadResult[]>([]), fileRef = useRef<HTMLInputElement>(null), uploadingRef = useRef(false)
  const monitor = useJob('data')
  const files = useQuery({ queryKey: ['data-list', root], queryFn: ({ signal }) => get('/api/data/list', { root }, { signal }) })
  const profile = useQuery({ queryKey: ['data-profile', selected], enabled: !!selected, queryFn: ({ signal }) => post('/api/data/profile', { path: selected }, { signal }) })
  const catalog = useQuery({ queryKey: ['data-catalog'], queryFn: () => get('/api/data/catalog') })
  async function uploadFiles(list: FileList | null) {
    if (!list?.length || uploadingRef.current) return
    uploadingRef.current = true; setUploading(true); setResults([])
    try {
      const next: UploadResult[] = []
      for (const file of Array.from(list)) {
        try { const result = await upload(file); next.push({ name: file.name, path: result.path }) }
        catch (error) { next.push({ name: file.name, error: errorMessage(error) }) }
        setResults([...next])
      }
      await files.refetch()
    } finally { uploadingRef.current = false; setUploading(false); if (fileRef.current) fileRef.current.value = '' }
  }
  async function start(body: Record<string, any>) { try { await monitor.start(body) } catch { /* visible error is owned by monitor */ } }
  const status = monitor.error ? 'Error: ' + monitor.error : monitor.job?.done ? monitor.job.code === 0 ? '✓ Job completed' : 'Error: job exited ' + monitor.job.code : ''
  const log = <><StatusText msg={status} /><LogBox lines={(monitor.job?.lines ?? []).slice(-600)} running={monitor.running} />{monitor.jobId && !monitor.job?.done && <Button onClick={monitor.stop} disabled={monitor.pending}>Stop job</Button>}{monitor.error && monitor.jobId && <Button onClick={() => monitor.select(null)}>Disconnect monitor</Button>}</>
  return <div style={{ padding: 28 }}><PageHeader title="Data" right={<Tabs tabs={[{ id: 'browse', label: 'Browse & Upload' }, { id: 'download', label: 'Download' }, { id: 'process', label: 'Process' }]} active={tab} onChange={setTab} />} />
    {tab === 'browse' && <div className="studio-split" style={{ display: 'grid', gridTemplateColumns: 'minmax(300px,1fr) 340px', gap: 14 }}>
      <Card><CardTitle>Files</CardTitle><Field label="Data root"><input className="input" value={root} onChange={event => setRoot(event.target.value)} /></Field>
        <div onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); uploadFiles(event.dataTransfer.files) }} style={{ border: '1px dashed var(--color-line)', padding: 18, margin: '12px 0' }}>
          <Button onClick={() => fileRef.current?.click()} disabled={uploading} icon={<Upload size={14} />}>{uploading ? 'Uploading…' : 'Choose files or drop them here'}</Button><input aria-label="Upload files" ref={fileRef} type="file" hidden multiple onChange={event => uploadFiles(event.target.files)} />
        </div>
        {results.length > 0 && <div aria-live="polite"><p>{results.filter(item => !item.error).length} uploaded, {results.filter(item => item.error).length} failed</p>{results.map((item, index) => <div key={index} style={{ color: item.error ? 'var(--color-neg)' : 'var(--color-pos)' }}>{item.name}: {item.error ?? item.path}</div>)}</div>}
        {files.error && <div role="alert">{errorMessage(files.error)}</div>}
        {!files.data?.files?.length && <Empty>No files found.</Empty>}
        {(files.data?.files ?? []).map((file: any) => <button aria-label={'Profile ' + file.path} type="button" className="btn" key={file.path} onClick={() => setSelected(file.path)} aria-pressed={selected === file.path} style={{ width: '100%', justifyContent: 'space-between', marginBottom: 5 }}><span>{file.path}</span><span>{fmtBytes(file.size)}</span></button>)}
      </Card>
      <Card><CardTitle>Profile {selected ? '· ' + selected : ''}</CardTitle>{profile.isFetching && <p>Profiling…</p>}{profile.error && <div role="alert">{errorMessage(profile.error)}</div>}{profile.data?.profile && <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(profile.data.profile, null, 2)}</pre>}{!selected && <Empty>Select a file.</Empty>}</Card>
    </div>}
    {tab === 'download' && <><div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 12 }}>{(catalog.data?.catalog ?? []).map((dataset: any) => <Card key={dataset.id}><CardTitle>{dataset.name}</CardTitle><p>{dataset.desc}</p><small>{dataset.size}</small><Button icon={<Download size={13} />} onClick={() => start({ type: 'download', dataset: dataset.id })} disabled={monitor.running}>Download</Button></Card>)}</div>{catalog.error && <div role="alert">{errorMessage(catalog.error)}</div>}{log}</>}
    {tab === 'process' && <div className="studio-split" style={{ display: 'grid', gridTemplateColumns: '320px minmax(300px,1fr)', gap: 14 }}><Card><CardTitle>Prepare dataset</CardTitle><Field label="Input paths (one per line)"><textarea className="textarea" rows={4} value={inputs} onChange={event => setInputs(event.target.value)} /></Field><Field label="Output directory"><input className="input" value={output} onChange={event => setOutput(event.target.value)} /></Field>
      <Field label="Mode"><select className="select" value={mode} onChange={event => setMode(event.target.value)}>{['auto','generative','chat','instruct','completion','code','fim','reasoning'].map(value => <option key={value}>{value}</option>)}</select></Field>
      <Field label="Tokenizer"><select className="select" value={tokenizer} onChange={event => setTokenizer(event.target.value)}>{['byte','byte-native','bpe','native-bpe','wordpiece','sentencepiece'].map(value => <option key={value}>{value}</option>)}</select></Field>
      <Field label="Sequence length"><input className="input" type="number" min={2} max={32768} value={seqLen} onChange={event => setSeqLen(+event.target.value)} /></Field>
      <Button icon={<Zap size={13} />} variant="primary" disabled={monitor.running || !inputs.trim() || !output.trim() || seqLen < 2} onClick={() => start({ type: 'prepare', input: inputs.split('\n').map(value => value.trim()).filter(Boolean), out: output, mode, tokenizer, seq_len: seqLen })}>Process</Button>
    </Card><Card>{log}</Card></div>}
  </div>
}
