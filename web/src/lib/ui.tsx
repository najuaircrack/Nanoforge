import type { ReactNode, CSSProperties } from 'react'
import { useId } from 'react'

/* Shared primitives — one place for card/button/field/etc.
   Styling lives in index.css component classes. */

export function PageHeader({ title, subtitle, right }: { title: string; subtitle?: ReactNode; right?: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 16, marginBottom: 24 }}>
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0 }}>{title}</h1>
        {subtitle != null && (
          <div style={{ color: 'var(--color-ink-2)', fontSize: 13, marginTop: 5 }}>{subtitle}</div>
        )}
      </div>
      {right}
    </div>
  )
}

export function Card({ children, pad = 20, className = '', style }: { children: ReactNode; pad?: number; className?: string; style?: CSSProperties }) {
  return <div className={`card ${className}`} style={{ padding: pad, ...style }}>{children}</div>
}

export function CardTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
      <div style={{ fontSize: 12, color: 'var(--color-ink-2)', fontWeight: 500, letterSpacing: '0.01em' }}>{children}</div>
      {right}
    </div>
  )
}

type BtnProps = {
  children: ReactNode; onClick?: () => void; disabled?: boolean
  variant?: 'default' | 'primary' | 'ghost'; icon?: ReactNode; style?: CSSProperties; title?: string
}
export function Button({ children, onClick, disabled, variant = 'default', icon, style, title }: BtnProps) {
  const cls = variant === 'primary' ? 'btn btn-primary' : variant === 'ghost' ? 'btn btn-ghost' : 'btn'
  return (
    <button className={cls} onClick={onClick} disabled={disabled} style={style} title={title}>
      {icon}{children}
    </button>
  )
}

export function Field({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span className="field-label">{label}</span>
      {children}
    </label>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div style={{ color: 'var(--color-ink-3)', fontSize: 13, padding: '10px 0' }}>{children}</div>
}

export function StatusText({ msg }: { msg: string }) {
  if (!msg) return null
  const ok = msg.startsWith('✓')
  const err = /error/i.test(msg)
  return <div style={{ fontSize: 12, color: ok ? 'var(--color-pos)' : err ? 'var(--color-neg)' : 'var(--color-ink-2)' }}>{msg}</div>
}

export function LogBox({ lines, running, minHeight = 220, maxHeight = 440 }: { lines: string[]; running?: boolean; minHeight?: number; maxHeight?: number }) {
  return (
    <div style={{
      background: '#08090c', border: '1px solid var(--color-line)', borderRadius: 'var(--radius-ui)',
      padding: 16, minHeight, maxHeight, overflow: 'auto',
      fontFamily: 'var(--font-mono)', fontSize: 12, lineHeight: 1.65, color: 'var(--color-ink-2)',
    }}>
      {running && (
        <div style={{ color: 'var(--color-pos)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 7 }}>
          <span className="live-dot">●</span> running
        </div>
      )}
      {lines.length === 0 && !running && <div style={{ color: 'var(--color-ink-3)' }}>Output will appear here.</div>}
      {lines.map((l, i) => <div key={i} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{l}</div>)}
    </div>
  )
}

/* ── Premium data-viz + control primitives ─────────────────────────────────── */

// Radial gauge — SVG arc, 270° sweep. value/max drives fill; color themable.
export function Gauge({
  value, max = 1, label, display, color = 'var(--color-accent)', size = 132, unit,
}: {
  value: number | null | undefined; max?: number; label: string
  display?: string; color?: string; size?: number; unit?: string
}) {
  const gid = useId()
  const frac = value == null || !isFinite(value) || max <= 0 ? 0 : Math.max(0, Math.min(1, value / max))
  const stroke = 10
  const r = (size - stroke) / 2 - 2
  const cx = size / 2
  const cy = size / 2
  const startAngle = 135          // degrees; sweep 270°
  const sweep = 270
  const polar = (a: number) => {
    const rad = (a * Math.PI) / 180
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)]
  }
  const arc = (a0: number, a1: number) => {
    const [x0, y0] = polar(a0)
    const [x1, y1] = polar(a1)
    const large = a1 - a0 > 180 ? 1 : 0
    return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
      <div style={{ position: 'relative', width: size, height: size * 0.82 }}>
        <svg width={size} height={size} style={{ position: 'absolute', top: 0, left: 0 }}>
          <defs>
            <linearGradient id={`g-${gid}`} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.55} />
              <stop offset="100%" stopColor={color} stopOpacity={1} />
            </linearGradient>
          </defs>
          <path d={arc(startAngle, startAngle + sweep)} fill="none"
            stroke="var(--color-surface-3)" strokeWidth={stroke} strokeLinecap="round" />
          {frac > 0 && (
            <path d={arc(startAngle, startAngle + sweep * frac)} fill="none"
              stroke={`url(#g-${gid})`} strokeWidth={stroke} strokeLinecap="round"
              style={{ transition: 'all .5s cubic-bezier(0.16,1,0.3,1)' }} />
          )}
        </svg>
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', paddingBottom: size * 0.14,
        }}>
          <div style={{ fontSize: 23, fontWeight: 700, fontFamily: 'var(--font-mono)', letterSpacing: '-0.03em', color: value != null ? 'var(--color-ink)' : 'var(--color-ink-3)' }}>
            {display ?? (value != null ? value.toFixed(0) : '—')}
          </div>
          {unit && <div style={{ fontSize: 10, color: 'var(--color-ink-3)', marginTop: 2 }}>{unit}</div>}
        </div>
      </div>
      <div style={{ fontSize: 11, color: 'var(--color-ink-2)', fontWeight: 500, letterSpacing: '0.03em', textTransform: 'uppercase' }}>{label}</div>
    </div>
  )
}

// Compact stat tile with optional accent + delta.
export function Stat({ label, value, color = 'var(--color-ink)', sub, mono = true }: {
  label: string; value: ReactNode; color?: string; sub?: ReactNode; mono?: boolean
}) {
  return (
    <div className="card" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 5 }}>
      <div style={{ fontSize: 10.5, color: 'var(--color-ink-3)', fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, fontFamily: mono ? 'var(--font-mono)' : 'inherit', color, letterSpacing: '-0.02em', lineHeight: 1.1 }}>{value}</div>
      {sub != null && <div style={{ fontSize: 11, color: 'var(--color-ink-3)' }}>{sub}</div>}
    </div>
  )
}

// Segmented tab strip.
export function Tabs<T extends string>({ tabs, active, onChange }: {
  tabs: { id: T; label: ReactNode }[]; active: T; onChange: (id: T) => void
}) {
  return (
    <div style={{ display: 'flex', gap: 2, background: 'var(--color-surface-2)', padding: 3, borderRadius: 10, border: '1px solid var(--color-line)' }}>
      {tabs.map(t => (
        <button key={t.id} onClick={() => onChange(t.id)} style={{
          padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
          fontSize: 13, fontWeight: 500, whiteSpace: 'nowrap',
          background: active === t.id ? 'var(--color-accent)' : 'transparent',
          color: active === t.id ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
          transition: 'background .12s, color .12s',
        }}>{t.label}</button>
      ))}
    </div>
  )
}

// On/off toggle switch.
export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: ReactNode }) {
  return (
    <button onClick={() => onChange(!checked)} style={{
      display: 'flex', alignItems: 'center', gap: 9, border: 'none', background: 'transparent',
      cursor: 'pointer', padding: 0, color: 'var(--color-ink)', fontSize: 13,
    }}>
      <span style={{
        width: 34, height: 20, borderRadius: 999, flexShrink: 0, position: 'relative',
        background: checked ? 'var(--color-accent)' : 'var(--color-surface-3)',
        border: `1px solid ${checked ? 'transparent' : 'var(--color-line-2)'}`,
        transition: 'background .15s',
      }}>
        <span style={{
          position: 'absolute', top: 2, left: checked ? 16 : 2,
          width: 14, height: 14, borderRadius: '50%',
          background: checked ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
          transition: 'left .15s',
        }} />
      </span>
      {label}
    </button>
  )
}

// Labeled range slider with live numeric readout.
export function Slider({ value, min, max, step = 1, onChange, format, label }: {
  value: number; min: number; max: number; step?: number
  onChange: (v: number) => void; format?: (v: number) => string; label?: string
}) {
  const row = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--color-ink)', minWidth: 54, textAlign: 'right' }}>
        {format ? format(value) : value}
      </span>
    </div>
  )
  if (!label) return row
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <span style={{ fontSize: 12, color: 'var(--color-ink-2)' }}>{label}</span>
      {row}
    </div>
  )
}

// Tiny inline sparkline (no axes) for dense metric tiles.
export function Sparkline({ data, color = 'var(--color-accent)', width = 120, height = 32 }: {
  data: number[]; color?: string; width?: number; height?: number
}) {
  const pts = data.filter(v => v != null && isFinite(v))
  if (pts.length < 2) return <div style={{ width, height, opacity: 0.3 }} />
  const min = Math.min(...pts), max = Math.max(...pts)
  const span = max - min || 1
  const path = pts.map((v, i) => {
    const x = (i / (pts.length - 1)) * width
    const y = height - ((v - min) / span) * (height - 4) - 2
    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <path d={path} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

// SVG architecture flow diagram — renders the transformer stack from a config.
export function ArchFlow({ cfg }: { cfg: Record<string, any> }) {
  const model = cfg?.model ?? {}
  const nLayers = model.n_layers ?? 0
  const box = (label: string, sub: string, accent = false) => (
    <div style={{
      padding: '9px 12px', borderRadius: 9, textAlign: 'center', minWidth: 120,
      background: accent ? 'color-mix(in srgb, var(--color-accent) 14%, var(--color-surface-2))' : 'var(--color-surface-2)',
      border: `1px solid ${accent ? 'var(--color-accent)' : 'var(--color-line-2)'}`,
    }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-ink)' }}>{label}</div>
      <div style={{ fontSize: 10.5, color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>{sub}</div>
    </div>
  )
  const arrow = <div style={{ color: 'var(--color-ink-3)', fontSize: 15, lineHeight: 1 }}>↓</div>
  const attn = model.attention_backend ?? 'sdpa'
  const ffn = model.ffn_type ?? model.activation ?? 'swiglu'
  const norm = model.normalization ?? 'rmsnorm'
  const pos = model.position_embedding ?? 'rope'
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 7 }}>
      {box('Token Embedding', `vocab ${model.vocab_size ?? '—'} · d ${model.d_model ?? '—'}`)}
      {arrow}
      <div style={{
        border: '1px dashed var(--color-line-2)', borderRadius: 12, padding: 14, width: '100%',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 7,
        background: 'color-mix(in srgb, var(--color-accent) 4%, transparent)',
      }}>
        <div style={{ fontSize: 10.5, color: 'var(--color-ink-3)', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
          × {nLayers} decoder blocks
        </div>
        {box(`Attention · ${attn}`, `${model.n_heads ?? '—'}h / ${model.n_kv_heads ?? model.n_heads ?? '—'}kv · ${pos}`, true)}
        {arrow}
        {box(`FFN · ${ffn}`, `mult ${model.ffn_hidden_mult ?? '—'} · ${norm}`, true)}
      </div>
      {arrow}
      {box('Final Norm + LM Head', `${model.task_type ?? 'causal_lm'}${model.tie_embeddings ? ' · tied' : ''}`)}
    </div>
  )
}

