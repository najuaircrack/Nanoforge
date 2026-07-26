import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Save, RotateCcw, FileCode2 } from 'lucide-react'
import { get, post } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Tabs, Toggle, Slider, StatusText } from '../lib/ui'

type FieldSchema = {
  name: string
  type: string
  default: any
  widget: 'select' | 'toggle' | 'number' | 'text'
  options?: string[]
  range?: { min: number; max: number; step: number }
}
type Section = { label: string; fields: FieldSchema[] }
type SchemaMap = Record<string, Section>

const SECTION_ORDER = ['model', 'training', 'data', 'inference']

// Human-friendly field labels + tooltips for the most important knobs.
const HELP: Record<string, string> = {
  d_model: 'Hidden size — the model width.',
  n_layers: 'Number of transformer blocks (depth).',
  n_heads: 'Attention heads. d_model must divide evenly.',
  n_kv_heads: 'KV heads for grouped-query attention (≤ n_heads).',
  vocab_size: 'Tokenizer vocabulary size.',
  max_seq_len: 'Maximum context length in tokens.',
  ffn_hidden_mult: 'FFN inner width as a multiple of d_model.',
  learning_rate: 'Peak learning rate after warmup.',
  batch_size: 'Global batch size (micro × grad-accum × world).',
  grad_accum_steps: 'Micro-batches accumulated per optimizer step.',
  max_steps: 'Total optimizer steps to train.',
  warmup_steps: 'Linear LR warmup steps.',
  seq_len: 'Packed sequence length for training samples.',
}

function labelize(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function coerce(field: FieldSchema, raw: any): any {
  if (field.widget === 'toggle') return Boolean(raw)
  if (field.widget === 'number') {
    if (raw === '' || raw == null) return null
    const n = Number(raw)
    return Number.isNaN(n) ? raw : n
  }
  return raw
}

function FieldRow({ field, value, onChange }: { field: FieldSchema; value: any; onChange: (v: any) => void }) {
  const help = HELP[field.name]
  const control = (() => {
    if (field.widget === 'toggle') {
      return <Toggle checked={Boolean(value)} onChange={onChange} />
    }
    if (field.widget === 'select') {
      return (
        <select className="select" value={value ?? ''} onChange={e => onChange(e.target.value)} style={{ maxWidth: 220 }}>
          {(field.options ?? []).map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      )
    }
    if (field.widget === 'number' && field.range) {
      const r = field.range
      const fractional = r.step < 1
      return (
        <div style={{ width: 240 }}>
          <Slider value={Number(value ?? 0)} min={r.min} max={r.max} step={r.step}
            onChange={onChange} format={v => fractional ? v.toFixed(3) : String(v)} />
        </div>
      )
    }
    if (field.widget === 'number') {
      return <input className="input" type="number" value={value ?? ''} onChange={e => onChange(e.target.value === '' ? null : Number(e.target.value))} style={{ maxWidth: 160 }} />
    }
    return <input className="input" value={value ?? ''} onChange={e => onChange(e.target.value)} style={{ maxWidth: 220 }} />
  })()

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, padding: '9px 0', borderBottom: '1px solid var(--color-line)' }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13, color: 'var(--color-ink)', fontWeight: 500 }}>{labelize(field.name)}</div>
        {help && <div style={{ fontSize: 11, color: 'var(--color-ink-3)', marginTop: 2 }}>{help}</div>}
        <div style={{ fontSize: 10.5, color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>{field.name}</div>
      </div>
      <div style={{ flexShrink: 0, display: 'flex', justifyContent: 'flex-end' }}>{control}</div>
    </div>
  )
}

export default function ConfigBuilder() {
  const [tab, setTab] = useState<string>('model')
  const [values, setValues] = useState<Record<string, Record<string, any>>>({})
  const [savePath, setSavePath] = useState('configs/custom.yaml')
  const [yamlText, setYamlText] = useState('')
  const [status, setStatus] = useState('')
  const [buildError, setBuildError] = useState('')

  const { data: schemaData } = useQuery({ queryKey: ['config-schema'], queryFn: () => get('/api/config/schema') })
  const schema: SchemaMap = schemaData?.schema ?? {}

  const { data: tmplData } = useQuery({ queryKey: ['cfg-templates'], queryFn: () => get('/api/config/templates') })
  const templates: string[] = tmplData?.templates ?? []

  // Seed defaults once the schema arrives.
  useEffect(() => {
    if (!schema || Object.keys(schema).length === 0 || Object.keys(values).length > 0) return
    const seed: Record<string, Record<string, any>> = {}
    for (const [sec, def] of Object.entries(schema)) {
      seed[sec] = {}
      for (const f of def.fields) seed[sec][f.name] = f.default
    }
    setValues(seed)
  }, [schema]) // eslint-disable-line react-hooks/exhaustive-deps

  // Debounced live build → canonical YAML + validation.
  const payload = useMemo(() => JSON.stringify(values), [values])
  useEffect(() => {
    if (Object.keys(values).length === 0) return
    const h = setTimeout(async () => {
      const r = await post('/api/config/build', { sections: values })
      if (r.ok) { setYamlText(r.yaml); setBuildError('') }
      else setBuildError(r.error)
    }, 350)
    return () => clearTimeout(h)
  }, [payload]) // eslint-disable-line react-hooks/exhaustive-deps

  function setField(sec: string, field: FieldSchema, raw: any) {
    setValues(v => ({ ...v, [sec]: { ...v[sec], [field.name]: coerce(field, raw) } }))
  }

  async function loadTemplate(path: string) {
    if (!path) return
    const r = await get('/api/config/load', { path })
    if (!r.ok) { setStatus(`Error: ${r.error}`); return }
    // Merge loaded values over current defaults.
    setValues(v => {
      const next = { ...v }
      for (const sec of SECTION_ORDER) {
        next[sec] = { ...(next[sec] ?? {}), ...(r.config[sec] ?? {}) }
      }
      return next
    })
    setSavePath(path)
    setStatus(`Loaded ${path}`)
  }

  function resetDefaults() {
    const seed: Record<string, Record<string, any>> = {}
    for (const [sec, def] of Object.entries(schema)) {
      seed[sec] = {}
      for (const f of def.fields) seed[sec][f.name] = f.default
    }
    setValues(seed)
    setStatus('Reset to defaults')
  }

  async function save() {
    setStatus('Saving…')
    const r = await post('/api/config/build', { sections: values, path: savePath })
    if (r.ok) setStatus(`✓ Saved to ${r.path}`)
    else setStatus(`Error: ${r.error}`)
  }

  const tabs = SECTION_ORDER.filter(s => schema[s]).map(s => ({ id: s, label: schema[s].label }))
  const active = schema[tab]

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Config Builder"
        subtitle="Full control over every model, training, data & inference parameter — validated live."
        right={
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <select className="select" defaultValue="" onChange={e => { loadTemplate(e.target.value); e.target.value = '' }} style={{ width: 200 }}>
              <option value="">Load template…</option>
              {templates.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <Button variant="ghost" onClick={resetDefaults} icon={<RotateCcw size={12} />}>Reset</Button>
          </div>
        }
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 460px', gap: 14 }}>
        {/* Editor */}
        <Card>
          <div style={{ marginBottom: 14 }}>
            <Tabs tabs={tabs} active={tab} onChange={setTab} />
          </div>
          {active
            ? <div>{active.fields.map(f => (
                <FieldRow key={f.name} field={f}
                  value={values[tab]?.[f.name]}
                  onChange={raw => setField(tab, f, raw)} />
              ))}</div>
            : <div style={{ color: 'var(--color-ink-3)', fontSize: 13 }}>Loading schema…</div>}
        </Card>

        {/* Live YAML preview */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <CardTitle right={<FileCode2 size={13} style={{ color: 'var(--color-ink-3)' }} />}>Live YAML</CardTitle>
            {buildError && (
              <div style={{ fontSize: 12, color: 'var(--color-neg)', background: 'color-mix(in srgb, var(--color-neg) 12%, transparent)', padding: '8px 10px', borderRadius: 8, fontFamily: 'var(--font-mono)' }}>
                {buildError}
              </div>
            )}
            <pre style={{
              margin: 0, background: 'var(--color-surface-2)', border: '1px solid var(--color-line)',
              borderRadius: 8, padding: '10px 12px', maxHeight: 520, overflow: 'auto',
              fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.6, color: 'var(--color-ink-2)',
            }}>{yamlText || '# adjust parameters to preview'}</pre>
          </Card>

          <Card style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', gap: 8 }}>
              <input className="input" value={savePath} onChange={e => setSavePath(e.target.value)} placeholder="configs/custom.yaml" style={{ flex: 1 }} />
              <Button variant="primary" onClick={save} disabled={!!buildError} icon={<Save size={13} />}>Save</Button>
            </div>
            <StatusText msg={status} />
            <div style={{ fontSize: 11, color: 'var(--color-ink-3)' }}>
              Saved configs appear in the Train tab. Launch with <span className="mono">Play</span> there or <span className="mono">nanoforge train --config {savePath}</span>.
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
