import { useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Save, Plus, RefreshCw } from 'lucide-react'
import { get, getText, post, errorMessage } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Empty, Field, StatusText } from '../lib/ui'
export default function Configs() {
  const [selected, setSelected] = useState(''), [yaml, setYaml] = useState(''), [savePath, setSavePath] = useState('configs/new.yaml'), [status, setStatus] = useState('')
  const [saving, setSaving] = useState(false), version = useRef(0)
  const query = useQuery({ queryKey: ['cfg-templates'], queryFn: () => get('/api/config/templates') })
  async function loadRaw(path: string) {
    const current = ++version.current; setStatus('Loading…')
    try { const text = await getText('/api/config/raw', { path }); if (current !== version.current) return; setYaml(text); setSavePath(path); setSelected(path); setStatus('') }
    catch (error) { if (current === version.current) setStatus('Error: ' + errorMessage(error)) }
  }
  async function saveConfig() {
    setSaving(true); setStatus('Saving…')
    try { const result = await post('/api/config/save', { path: savePath, yaml_text: yaml }); setStatus('✓ Saved to ' + result.path); await query.refetch() }
    catch (error) { setStatus('Error: ' + errorMessage(error)) } finally { setSaving(false) }
  }
  return <div style={{ padding: 28 }}><PageHeader title="Configs" right={<Button onClick={() => query.refetch()} icon={<RefreshCw size={12} />}>Refresh</Button>} />
    {query.error && <div role="alert">{errorMessage(query.error)}</div>}
    <div className="studio-split" style={{ display: 'grid', gridTemplateColumns: '240px minmax(300px,1fr)', gap: 14 }}>
      <Card><CardTitle>Templates</CardTitle>{(query.data?.templates ?? []).map((path: string) => <button className="btn" key={path} onClick={() => loadRaw(path)} aria-pressed={selected === path} style={{ display: 'block', marginBottom: 6 }}>{path}</button>)}
        {!query.data?.templates?.length && <Empty>No configs found.</Empty>}
        <Button icon={<Plus size={12} />} onClick={() => { version.current++; setYaml('model:\n  d_model: 256\n  n_layers: 4\n  n_heads: 4\n'); setSavePath('configs/new.yaml'); setSelected(''); setStatus('') }}>New config</Button>
      </Card>
      <Card><Field label="Config path"><input className="input" value={savePath} onChange={event => setSavePath(event.target.value)} /></Field>
        <Field label="YAML configuration"><textarea className="textarea" value={yaml} onChange={event => setYaml(event.target.value)} rows={22} /></Field>
        <Button icon={<Save size={13} />} variant="primary" onClick={saveConfig} disabled={saving || !yaml || !savePath.trim()}>Save</Button><StatusText msg={status} />
      </Card>
    </div>
  </div>
}
