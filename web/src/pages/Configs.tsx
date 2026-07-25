import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Save, Plus, RefreshCw } from 'lucide-react'
import { get, post } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Empty } from '../lib/ui'

export default function Configs() {
  const [selected, setSelected] = useState('')
  const [yaml, setYaml] = useState('')
  const [savePath, setSavePath] = useState('configs/new.yaml')
  const [status, setStatus] = useState('')

  const { data, refetch } = useQuery({ queryKey: ['cfg-templates'], queryFn: () => get('/api/config/templates') })
  const templates: string[] = data?.templates ?? []

  async function loadRaw(path: string) {
    setStatus('')
    try {
      const r = await fetch(`/api/config/raw?path=${encodeURIComponent(path)}`)
      const text = await r.text()
      setYaml(text); setSavePath(path); setSelected(path)
    } catch {
      const r = await get('/api/config/load', { path })
      if (r.ok) {
        const lines: string[] = []
        for (const [section, vals] of Object.entries(r.config as Record<string, any>)) {
          lines.push(`${section}:`)
          for (const [k, v] of Object.entries(vals as Record<string, any>))
            lines.push(`  ${k}: ${v === null ? 'null' : JSON.stringify(v)}`)
          lines.push('')
        }
        setYaml(lines.join('\n')); setSavePath(path); setSelected(path)
      } else setStatus(`Error: ${r.error}`)
    }
  }

  async function saveConfig() {
    if (!savePath || !yaml) return
    setStatus('Saving…')
    const r = await post('/api/config/save', { path: savePath, yaml_text: yaml })
    if (r.ok) { setStatus(`✓ Saved to ${r.path}`); refetch() }
    else setStatus(`Error: ${r.error}`)
  }

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Configs"
        right={
          <Button variant="ghost" onClick={() => refetch()} icon={<RefreshCw size={12} />} style={{ fontSize: 12, padding: '4px 10px' }}>
            Refresh
          </Button>
        }
      />

      <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr', gap: 14 }}>
        <Card>
          <CardTitle>Templates</CardTitle>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginBottom: 10 }}>
            {templates.length === 0 && <Empty>No configs found in configs/</Empty>}
            {templates.map(t => (
              <button key={t} onClick={() => loadRaw(t)} style={{
                textAlign: 'left', padding: '7px 10px', borderRadius: 8, border: 'none', cursor: 'pointer',
                background: selected === t ? 'var(--color-surface-3)' : 'transparent',
                color: selected === t ? 'var(--color-ink)' : 'var(--color-ink-2)',
                fontSize: 12, fontFamily: 'var(--font-mono)',
              }}>{t}</button>
            ))}
          </div>
          <button onClick={() => { setYaml('model:\n  d_model: 256\n  n_layers: 4\n  n_heads: 4\n'); setSavePath('configs/new.yaml'); setSelected('') }}
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '7px', borderRadius: 8, border: '1px dashed var(--color-line-2)', background: 'transparent', color: 'var(--color-ink-2)', cursor: 'pointer', fontSize: 12 }}>
            <Plus size={12} /> New config
          </button>
        </Card>

        <Card style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input value={savePath} onChange={e => setSavePath(e.target.value)} className="input" placeholder="configs/my-model.yaml" style={{ flex: 1 }} />
            <Button onClick={saveConfig} disabled={!yaml || !savePath} variant="primary" icon={<Save size={13} />}>
              Save
            </Button>
          </div>
          {status && (
            <div style={{ fontSize: 12, color: status.startsWith('✓') ? 'var(--color-pos)' : 'var(--color-neg)' }}>{status}</div>
          )}
          <textarea value={yaml} onChange={e => setYaml(e.target.value)}
            style={{
              flex: 1, minHeight: 520, resize: 'vertical',
              background: 'var(--color-surface-2)', border: '1px solid var(--color-line)',
              borderRadius: 8, padding: '10px 12px', color: 'var(--color-ink)',
              fontFamily: 'var(--font-mono)', fontSize: 12, lineHeight: 1.65,
            }}
            placeholder="# YAML config will appear here&#10;# Select a template or create a new config" />
        </Card>
      </div>
    </div>
  )
}
