import { useEffect, useId, useRef, useState } from 'react'
import type { FieldSchema } from './contracts'
import { fieldKind, labelize, nullable, parseField } from './contracts'
export function ConfigField({ field, value, onChange, onValidity }: { field: FieldSchema; value: any; onChange: (value: any) => void; onValidity?: (valid: boolean) => void }) {
  const id = useId(), kind = fieldKind(field), optional = nullable(field)
  const format = (item: any) => kind === 'array' || kind === 'object' ? (item == null ? '' : JSON.stringify(item, null, 2)) : item == null ? '' : String(item)
  const [draft, setDraft] = useState(() => format(value)), [error, setError] = useState('')
  const committed = useRef(value)
  useEffect(() => { if (!Object.is(value, committed.current)) { committed.current = value; setDraft(format(value)); setError('') } }, [value])
  function change(raw: string) { setDraft(raw); try { const next = parseField(field, raw); committed.current = next; setError(''); onValidity?.(true); onChange(next) } catch (failure) { setError((failure as Error).message); onValidity?.(false) } }
  let control
  if (kind === 'array' || kind === 'object') control = <textarea id={id} className="textarea" rows={4} value={draft} onChange={event => change(event.target.value)} aria-invalid={!!error} aria-describedby={error ? id + '-error' : undefined} placeholder={optional ? 'null / automatic' : kind === 'array' ? '[]' : '{}'} />
  else if (kind === 'boolean') control = <select id={id} className="select" value={value == null ? 'null' : String(value)} onChange={event => change(event.target.value)}>{optional && <option value="null">Automatic</option>}<option value="true">True</option><option value="false">False</option></select>
  else if (kind === 'select') control = <select id={id} className="select" value={value ?? ''} onChange={event => change(event.target.value)}>{optional && <option value="">Automatic</option>}{field.options?.map(option => <option key={option} value={option}>{option}</option>)}</select>
  else control = <input id={id} className="input" type={kind === 'number' ? 'number' : 'text'} value={draft} step={kind === 'number' && field.type.startsWith('int') ? 1 : 'any'} onChange={event => change(event.target.value)} aria-invalid={!!error} placeholder={optional ? 'Automatic' : undefined} />
  return <div style={{ padding: '12px 0', borderBottom: '1px solid var(--color-line)', display: 'grid', gridTemplateColumns: 'minmax(130px, 1fr) minmax(180px, 1.2fr)', gap: 16 }}>
    <label htmlFor={id} style={{ fontSize: 13 }}>{labelize(field.name)}<small aria-hidden="true" style={{ display: 'block', color: 'var(--color-ink-3)', marginTop: 4 }}>{field.type}</small></label>
    <div>{control}{error && <div id={id + '-error'} role="alert" style={{ color: 'var(--color-neg)', fontSize: 12, marginTop: 6 }}>{error}</div>}</div>
  </div>
}
