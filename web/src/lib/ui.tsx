import type { ReactNode, CSSProperties } from 'react'

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
