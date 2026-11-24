export type FieldSchema = { name: string; type: string; default?: any; widget: string; options?: string[]; range?: { min: number; max: number; step: number }; nullable?: boolean }
export function fieldKind(field: FieldSchema): 'array' | 'object' | 'boolean' | 'number' | 'select' | 'text' {
  const type = field.type.toLowerCase()
  if (/^(list|tuple)\[/.test(type) || Array.isArray(field.default)) return 'array'
  if (/dict\[|moeconfig/.test(type)) return 'object'
  if (field.options?.length) return 'select'
  if (type.includes('bool')) return 'boolean'
  if (type.includes('int') || type.includes('float')) return 'number'
  return 'text'
}
export function nullable(field: FieldSchema) { return field.nullable ?? field.type.includes('None') }
export function labelize(name: string) { return name.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase()) }
export function parseField(field: FieldSchema, raw: string): any {
  if ((raw.trim() === '' || raw.trim() === 'null') && nullable(field)) return null
  const kind = fieldKind(field)
  if (kind === 'array' || kind === 'object') {
    let value: any
    try { value = JSON.parse(raw) } catch { throw new Error('Enter valid JSON') }
    if (kind === 'array') {
      if (!Array.isArray(value)) throw new Error('Expected a JSON array')
      if (field.type.startsWith('tuple[') && value.length !== field.type.split(',').length) throw new Error('Incorrect tuple length')
      if (/^(list|tuple)\[(float|int)/.test(field.type) && value.some(item => typeof item !== 'number' || !Number.isFinite(item))) throw new Error('Expected numeric array entries')
      if (field.type.startsWith('list[str]') && value.some(item => typeof item !== 'string')) throw new Error('Expected string array entries')
    } else if (value === null || Array.isArray(value) || typeof value !== 'object') throw new Error('Expected a JSON object')
    return value
  }
  if (kind === 'boolean') { if (raw === 'true') return true; if (raw === 'false') return false; throw new Error('Choose true, false or automatic') }
  if (kind === 'number') { const number = Number(raw); if (raw.trim() === '' || !Number.isFinite(number)) throw new Error('Enter a finite number'); if (field.type.startsWith('int') && !Number.isInteger(number)) throw new Error('Enter a whole number'); return number }
  return raw
}
