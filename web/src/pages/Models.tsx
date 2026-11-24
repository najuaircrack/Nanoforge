import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Box, Upload, RefreshCw } from 'lucide-react'
import { get, post, fmtBytes, errorMessage } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Field, Empty } from '../lib/ui'

export default function Models() {
  const [importSrc, setImportSrc] = useState('')
  const [importName, setImportName] = useState('')
  const [importBackend, setImportBackend] = useState('')
  const [status, setStatus] = useState('')
  const [pending, setPending] = useState(false)
  const navigate = useNavigate()

  const { data: modelData, refetch } = useQuery({ queryKey: ['models'], queryFn: () => get('/api/model/list') })
  const { data: importData, refetch: refetchImports } = useQuery({ queryKey: ['imports'], queryFn: () => get('/api/model/imports') })
  const models: any[] = (modelData?.models ?? []).map((model: any) => ({ ...model, path: model.ref ?? model.path ?? model.name }))
  const imports: any[] = importData?.models ?? []

  async function importModel() {
    if (!importSrc.trim() || !importName.trim() || pending) return
    setPending(true); setStatus('Importing…')
    try { const result = await post('/api/model/import', { source: importSrc.trim(), name: importName.trim(), backend: importBackend || undefined }); setStatus('✓ Imported ' + result.entry?.name); await refetchImports(); await refetch() }
    catch (error) { setStatus('Error: ' + errorMessage(error)) } finally { setPending(false) }
  }
  async function loadModel(ref: string) {
    setPending(true); setStatus('Loading…')
    try { await post('/api/chat/load', { path: ref }); navigate('/chat') }
    catch (error) { setStatus('Error: ' + errorMessage(error)) } finally { setPending(false) }
  }

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Models"
        right={
          <Button variant="ghost" onClick={() => { refetch(); refetchImports() }} icon={<RefreshCw size={12} />} style={{ fontSize: 12, padding: '4px 10px' }}>
            Refresh
          </Button>
        }
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 14 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card>
            <CardTitle right={
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-ink-3)' }}>{models.length} found</span>
            }>
              Available models
            </CardTitle>
            {models.length === 0
              ? <Empty>No checkpoints found in runs/ or models/</Empty>
              : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {models.map(m => (
                    <div key={m.path} style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      padding: '8px 10px', background: 'var(--color-surface-2)',
                      border: '1px solid var(--color-line)', borderRadius: 8,
                    }}>
                      <Box size={14} style={{ color: 'var(--color-accent)', flexShrink: 0 }} />
                      <span className="mono" style={{ flex: 1, fontSize: 12, color: 'var(--color-ink-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.path}</span>
                      <span style={{ fontSize: 11, color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)' }}>{fmtBytes(m.size)}</span>
                      <Button disabled={pending} onClick={() => loadModel(m.ref ?? m.path ?? m.name)}>Load</Button>
                    </div>
                  ))}
                </div>
              )
            }
          </Card>

          <Card>
            <CardTitle>Imported models</CardTitle>
            {imports.length === 0
              ? <Empty>No imported models. Use the import panel.</Empty>
              : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {imports.map((m: any) => (
                    <div key={m.name} style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      padding: '8px 10px', background: 'var(--color-surface-2)',
                      border: '1px solid var(--color-line)', borderRadius: 8,
                    }}>
                      <Box size={14} style={{ color: 'var(--color-pos)', flexShrink: 0 }} />
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--color-ink)' }}>{m.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)' }}>{m.backend} · {m.format} · {m.source}</div>
                      </div>
                      <Button disabled={pending} onClick={() => loadModel(m.ref ?? m.name)}>Load</Button>
                    </div>
                  ))}
                </div>
              )
            }
          </Card>
        </div>

        <Card>
          <CardTitle>
            <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}><Upload size={14} /> Import external model</span>
          </CardTitle>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <Field label="Source (path or HuggingFace ID)">
              <input value={importSrc} onChange={e => setImportSrc(e.target.value)} className="input" placeholder="mistralai/Mistral-7B-v0.1" />
            </Field>
            <Field label="Name">
              <input value={importName} onChange={e => setImportName(e.target.value)} className="input" placeholder="mistral-7b" />
            </Field>
            <Field label="Backend (auto-detected if blank)">
              <select value={importBackend} onChange={e => setImportBackend(e.target.value)} className="select">
                <option value="">auto</option>
                <option value="transformers">transformers</option>
                <option value="llama_cpp">llama_cpp</option>
                <option value="onnxruntime">onnxruntime</option>
              </select>
            </Field>
            <Button onClick={importModel} disabled={pending || !importSrc.trim() || !importName.trim()} variant="primary" icon={<Upload size={13} />}>
              Import
            </Button>
            {status && <div role="status" aria-live="polite" style={{ fontSize: 12, color: status.startsWith('✓') ? 'var(--color-pos)' : 'var(--color-neg)' }}>{status}</div>}
          </div>
        </Card>
      </div>
    </div>
  )
}
