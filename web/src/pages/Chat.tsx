import { useEffect, useRef, useState } from 'react'
import { Send, Loader2, Bot, ChevronDown } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { useQuery } from '@tanstack/react-query'
import { get, post } from '../lib/api'

type Msg = { role: 'user' | 'assistant'; content: string }

const STORAGE_KEY = 'nanoforge_loaded_model'

export default function Chat() {
  const [loaded, setLoaded] = useState(() => localStorage.getItem(STORAGE_KEY) ?? '')
  const [loading, setLoading] = useState(false)
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [showPicker, setShowPicker] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)

  const { data: modelData } = useQuery({ queryKey: ['models'], queryFn: () => get('/api/model/list') })
  const models: any[] = modelData?.models ?? []

  // Restore loaded model state from server on mount
  useEffect(() => {
    get('/api/chat/status').then(r => {
      if (r.loaded) { setLoaded(r.loaded); localStorage.setItem(STORAGE_KEY, r.loaded) }
    }).catch(() => {})
  }, [])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  function resizeTextarea() {
    const el = taRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 180) + 'px'
  }

  async function loadModel(path: string) {
    if (!path || path === loaded) return
    setLoading(true); setStatus('Loading…')
    const r = await post('/api/chat/load', { path })
    setLoading(false)
    if (r.ok) { setLoaded(path); localStorage.setItem(STORAGE_KEY, path); setStatus('') }
    else setStatus(`Error: ${r.error}`)
    setShowPicker(false)
  }

  async function send() {
    if (!input.trim() || busy || !loaded) return
    const text = input.trim()
    const history: Msg[] = [...msgs, { role: 'user', content: text }]
    setMsgs(history)
    setInput(''); setBusy(true)
    if (taRef.current) taRef.current.style.height = 'auto'

    // Add empty assistant message to stream into
    setMsgs(m => [...m, { role: 'assistant', content: '' }])

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: history.map(m => [m.role, m.content]),
          max_new_tokens: 512,
        }),
      })
      const reader = res.body!.getReader()
      const dec = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const msg = JSON.parse(line.slice(6))
          if (msg.type === 'chunk') {
            setMsgs(m => {
              const copy = [...m]
              copy[copy.length - 1] = { role: 'assistant', content: copy[copy.length - 1].content + msg.text }
              return copy
            })
          } else if (msg.type === 'error') {
            setMsgs(m => {
              const copy = [...m]
              copy[copy.length - 1] = { role: 'assistant', content: `[error: ${msg.text}]` }
              return copy
            })
          }
        }
      }
    } catch (e: any) {
      setMsgs(m => {
        const copy = [...m]
        copy[copy.length - 1] = { role: 'assistant', content: `[error: ${e.message}]` }
        return copy
      })
    }
    setBusy(false)
  }

  const modelName = loaded ? loaded.split(/[\\/]/).pop() : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100dvh', background: 'var(--color-bg)' }}
      onClick={() => setShowPicker(false)}>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '40px 20px 20px' }}>
        <div style={{ maxWidth: 680, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 0 }}>

          {msgs.length === 0 && (
            <div style={{ textAlign: 'center', marginTop: 100 }}>
              <div style={{
                width: 52, height: 52, borderRadius: 16, background: 'var(--color-surface-2)',
                border: '1px solid var(--color-line)', display: 'flex', alignItems: 'center',
                justifyContent: 'center', margin: '0 auto 18px',
              }}>
                <Bot size={24} style={{ color: 'var(--color-accent)' }} />
              </div>
              <div style={{ fontSize: 20, fontWeight: 600, color: 'var(--color-ink)', marginBottom: 8 }}>
                {modelName ?? 'No model loaded'}
              </div>
              <div style={{ fontSize: 14, color: 'var(--color-ink-3)', lineHeight: 1.6 }}>
                {loaded
                  ? 'Ask anything. Enter to send, Shift+Enter for a new line.'
                  : 'Pick a checkpoint from the composer below to begin.'}
              </div>
            </div>
          )}

          {msgs.map((m, i) => (
            <div key={i} style={{
              display: 'flex', gap: 12, padding: '18px 0',
              borderBottom: i < msgs.length - 1 ? '1px solid var(--color-line)' : 'none',
              alignItems: 'flex-start',
              flexDirection: m.role === 'user' ? 'row-reverse' : 'row',
            }}>
              <div style={{
                width: 28, height: 28, borderRadius: 8, flexShrink: 0, marginTop: 2,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 700,
                background: m.role === 'user' ? 'var(--color-accent)' : 'var(--color-surface-3)',
                color: m.role === 'user' ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
                border: '1px solid var(--color-line)',
              }}>
                {m.role === 'user' ? 'U' : <Bot size={13} />}
              </div>
              <div style={{
                fontSize: 15, lineHeight: 1.75, color: 'var(--color-ink)',
                maxWidth: '88%',
                ...(m.role === 'user' ? {
                  background: 'var(--color-surface-2)',
                  border: '1px solid var(--color-line)',
                  borderRadius: '14px 14px 4px 14px',
                  padding: '10px 14px',
                } : { padding: '2px 0' }),
              }}>
                {m.content
                  ? <ReactMarkdown>{m.content}</ReactMarkdown>
                  : busy && i === msgs.length - 1
                    ? <span style={{ display: 'inline-flex', gap: 4, alignItems: 'center', paddingTop: 6 }}>
                        {[0,1,2].map(j => (
                          <span key={j} style={{
                            width: 6, height: 6, borderRadius: '50%', background: 'var(--color-ink-3)',
                            display: 'inline-block',
                            animation: `blink 1.2s ease-in-out ${j*0.2}s infinite`,
                          }} />
                        ))}
                      </span>
                    : null
                }
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>
      </div>

      {/* Composer */}
      <div style={{ padding: '10px 20px 24px', flexShrink: 0 }}>
        <div style={{ maxWidth: 680, margin: '0 auto' }}>
          <div style={{
            background: 'var(--color-surface-2)', border: '1px solid var(--color-line-2)',
            borderRadius: 16, padding: '10px 12px 10px 16px',
            boxShadow: '0 4px 24px rgba(0,0,0,0.25)',
            display: 'flex', flexDirection: 'column', gap: 8,
          }}>
            {/* Text input */}
            <textarea ref={taRef} value={input}
              onChange={e => { setInput(e.target.value); resizeTextarea() }}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
              placeholder={loaded ? 'Message…' : 'Select a model below, then message…'}
              disabled={!loaded || busy} rows={1}
              style={{
                border: 'none', outline: 'none', background: 'transparent', resize: 'none',
                color: 'var(--color-ink)', fontFamily: 'var(--font-sans)', fontSize: 15,
                lineHeight: 1.6, width: '100%', maxHeight: 180, overflowY: 'auto',
              }}
            />

            {/* Bottom bar: model picker + send */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              {/* Model pill — Claude.ai style */}
              <div style={{ position: 'relative' }} onClick={e => e.stopPropagation()}>
                <button onClick={() => setShowPicker(s => !s)} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '5px 10px', borderRadius: 20,
                  border: '1px solid var(--color-line-2)',
                  background: showPicker ? 'var(--color-surface-3)' : 'transparent',
                  color: loaded ? 'var(--color-ink-2)' : 'var(--color-ink-3)',
                  cursor: 'pointer', fontSize: 12, fontFamily: 'var(--font-mono)',
                  maxWidth: 260, overflow: 'hidden',
                }}>
                  {loading
                    ? <Loader2 size={11} className="spin" />
                    : <span style={{ width: 7, height: 7, borderRadius: '50%', background: loaded ? 'var(--color-pos)' : 'var(--color-ink-3)', flexShrink: 0 }} />
                  }
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {modelName ?? 'Select model'}
                  </span>
                  <ChevronDown size={11} style={{ flexShrink: 0 }} />
                </button>

                {showPicker && (
                  <div style={{
                    position: 'absolute', bottom: 'calc(100% + 8px)', left: 0, zIndex: 50,
                    background: 'var(--color-surface-2)', border: '1px solid var(--color-line-2)',
                    borderRadius: 12, padding: 6, minWidth: 280, maxWidth: 380,
                    maxHeight: 260, overflowY: 'auto',
                    boxShadow: '0 -8px 32px rgba(0,0,0,0.4)',
                  }}>
                    {models.length === 0
                      ? <div style={{ padding: '8px 10px', fontSize: 12, color: 'var(--color-ink-3)' }}>No checkpoints found in runs/</div>
                      : models.map(m => (
                        <button key={m.path} onClick={() => loadModel(m.path)} style={{
                          display: 'block', width: '100%', textAlign: 'left',
                          padding: '8px 10px', borderRadius: 8, border: 'none', cursor: 'pointer',
                          background: m.path === loaded ? 'var(--color-surface-3)' : 'transparent',
                          color: m.path === loaded ? 'var(--color-ink)' : 'var(--color-ink-2)',
                          fontSize: 12, fontFamily: 'var(--font-mono)',
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        }}>
                          {m.path}
                        </button>
                      ))
                    }
                  </div>
                )}
              </div>

              {/* Send button */}
              <button onClick={send} disabled={!loaded || busy || !input.trim()} style={{
                width: 34, height: 34, borderRadius: 10, border: 'none', flexShrink: 0,
                background: loaded && input.trim() && !busy ? 'var(--color-accent)' : 'var(--color-surface-3)',
                color: loaded && input.trim() && !busy ? 'var(--color-accent-ink)' : 'var(--color-ink-3)',
                cursor: loaded && input.trim() && !busy ? 'pointer' : 'default',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                transition: 'background 0.15s, color 0.15s',
              }}>
                {busy ? <Loader2 size={14} className="spin" /> : <Send size={14} />}
              </button>
            </div>
          </div>

          {status && <div style={{ fontSize: 12, color: 'var(--color-neg)', marginTop: 6, paddingLeft: 4 }}>{status}</div>}
        </div>
      </div>

      <style>{`
        @keyframes blink {
          0%, 80%, 100% { opacity: 0.25; transform: scale(0.75); }
          40% { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  )
}
