import { useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Save, RotateCcw } from 'lucide-react'
import { get, post, errorMessage } from '../lib/api'
import { Card, CardTitle, PageHeader, Button, Tabs, StatusText, Field } from '../lib/ui'
import { ConfigField } from '../lib/ConfigField'
import type { FieldSchema } from '../lib/contracts'
type Schema = Record<string, { label: string; fields: FieldSchema[] }>
export default function ConfigBuilder() {
  const [tab, setTab] = useState('model'), [values, setValues] = useState<Record<string, Record<string, any>>>({})
  const [savePath, setSavePath] = useState('configs/custom.yaml'), [yamlText, setYamlText] = useState('')
  const [status, setStatus] = useState(''), [buildError, setBuildError] = useState(''), [validated, setValidated] = useState('')
  const [invalid, setInvalid] = useState<Record<string, boolean>>({}), [saving, setSaving] = useState(false)
  const templateVersion = useRef(0), client = useQueryClient()
  const schemaQuery = useQuery({ queryKey: ['config-schema'], queryFn: () => get('/api/config/schema') })
  const schema: Schema = schemaQuery.data?.schema ?? {}
  const templateQuery = useQuery({ queryKey: ['cfg-templates'], queryFn: () => get('/api/config/templates') })
  function defaults() { return Object.fromEntries(Object.entries(schema).map(([section, def]) => [section, Object.fromEntries(def.fields.map(field => [field.name, field.default]))])) }
  useEffect(() => { if (Object.keys(schema).length) setValues(current => Object.keys(current).length ? current : defaults()) }, [schemaQuery.data])
  const payload = JSON.stringify(values)
  useEffect(() => {
    if (!Object.keys(values).length) return
    const controller = new AbortController()
    setValidated(''); setBuildError('')
    const timer = setTimeout(async () => {
      try {
        const result = await post('/api/config/build', { sections: JSON.parse(payload) }, { signal: controller.signal })
        if (!controller.signal.aborted) { setYamlText(result.yaml); setValidated(payload); setBuildError('') }
      } catch (error) { if (!controller.signal.aborted) { setBuildError(errorMessage(error)); setYamlText('') } }
    }, 350)
    return () => { clearTimeout(timer); controller.abort() }
  }, [payload])
  async function loadTemplate(path: string) {
    if (!path) return
    const version = ++templateVersion.current
    try {
      const result = await get('/api/config/load', { path })
      if (version !== templateVersion.current) return
      const seed = defaults()
      for (const [section, sectionValues] of Object.entries(result.config)) seed[section] = { ...(seed[section] ?? {}), ...(sectionValues as Record<string, any>) }
      setValues(seed); setInvalid({}); setSavePath(path); setStatus('Loaded ' + path)
    } catch (error) { if (version === templateVersion.current) setStatus('Error: ' + errorMessage(error)) }
  }
  async function save() {
    setSaving(true); setStatus('Saving…')
    try { const result = await post('/api/config/build', { sections: values, path: savePath }); setStatus('✓ Saved to ' + result.path); await client.invalidateQueries({ queryKey: ['cfg-templates'] }) }
    catch (error) { setStatus('Error: ' + errorMessage(error)) } finally { setSaving(false) }
  }
  const tabs = Object.entries(schema).map(([id, section]) => ({ id, label: section.label }))
  const pending = validated !== payload
  return <div style={{ padding: 28 }}>
    <PageHeader title="Config Builder" subtitle="Typed parameters, structured JSON and server validation."
      right={<div style={{ display: 'flex', gap: 8 }}>
        <select aria-label="Load template" className="select" defaultValue="" onChange={event => { loadTemplate(event.target.value); event.target.value = '' }}><option value="">Load template…</option>{(templateQuery.data?.templates ?? []).map((path: string) => <option key={path}>{path}</option>)}</select>
        <Button icon={<RotateCcw size={12} />} onClick={() => { templateVersion.current++; setValues(defaults()); setInvalid({}); setStatus('Reset to defaults') }}>Reset</Button>
      </div>} />
    {schemaQuery.error && <div role="alert">Error: {errorMessage(schemaQuery.error)}</div>}
    <div className="studio-split" style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) minmax(260px, 420px)', gap: 14 }}>
      <Card><Tabs tabs={tabs} active={tab} onChange={setTab} />
        {(schema[tab]?.fields ?? []).map(field => <ConfigField key={tab + '.' + field.name} field={field} value={values[tab]?.[field.name]}
          onValidity={valid => setInvalid(current => ({ ...current, [tab + '.' + field.name]: !valid }))}
          onChange={value => setValues(current => ({ ...current, [tab]: { ...current[tab], [field.name]: value } }))} />)}
      </Card>
      <div><Card><CardTitle>Canonical YAML {pending && !buildError ? '· validating…' : ''}</CardTitle>
        {buildError && <div role="alert" style={{ color: 'var(--color-neg)' }}>{buildError}</div>}
        <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 520, overflow: 'auto', fontSize: 12 }}>{pending ? 'Waiting for current values to validate.' : yamlText}</pre>
        <Field label="Save config path"><input className="input" value={savePath} onChange={event => setSavePath(event.target.value)} /></Field>
        <Button variant="primary" icon={<Save size={13} />} onClick={save} disabled={saving || pending || !!buildError || Object.values(invalid).some(Boolean) || !savePath.trim()}>Save</Button><StatusText msg={status} />
      </Card></div>
    </div>
  </div>
}
