import { useEffect, useRef, useState } from 'react'
import { Send, Loader2, Bot, ChevronDown, Database, Sliders, Sparkles } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { useQuery } from '@tanstack/react-query'
import { get, post, authHeaders, consumeSSE, errorMessage } from '../lib/api'
import { Slider } from '../lib/ui'

type Msg = { role: 'user' | 'assistant'; content: string; source?: string; error?: boolean }

const STORAGE_KEY = 'nanoforge_loaded_model'

export default function Chat() {
  const [loaded, setLoaded] = useState('')
  const [loading, setLoading] = useState(true)
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [showPicker, setShowPicker] = useState(false)
  // Generation params
  const [maxTokens, setMaxTokens] = useState(512)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(50)
  const [showParams, setShowParams] = useState(false)
  // RAG index state
  const [ragLoaded, setRagLoaded] = useState('')
  const [ragPath, setRagPath] = useState('')
  const [ragBusy, setRagBusy] = useState(false)
  const [showRag, setShowRag] = useState(false)
  // Personal-facts index state (retrieve + templated answer, bypasses the model)
  const [factsLoaded, setFactsLoaded] = useState('')
  const [factsPath, setFactsPath] = useState('')
  const [factsBusy, setFactsBusy] = useState(false)
  const [showFacts, setShowFacts] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)
  const streamAbort = useRef<AbortController | null>(null)

  const { data: modelData } = useQuery({ queryKey: ['models'], queryFn: () => get('/api/model/list') })
  const { data: importData } = useQuery({ queryKey: ['imports'], queryFn: () => get('/api/model/imports') })
  const models: any[] = [...(modelData?.models ?? []).map((model: any) => ({ ...model, path: model.ref ?? model.path ?? model.name })), ...(importData?.models ?? []).map((model: any) => ({ ...model, path: model.ref ?? model.name, kind: 'imported' }))].filter((model, index, list) => list.findIndex(other => other.path === model.path) === index)

  // Server state is authoritative, including an empty engine after restart.
  useEffect(() => {
    const controller = new AbortController()
    get('/api/chat/status', undefined, { signal: controller.signal }).then(result => {
      setLoaded(result.loaded ?? '')
      if (result.loaded) localStorage.setItem(STORAGE_KEY, result.loaded)
      else localStorage.removeItem(STORAGE_KEY)
    }).catch(error => { if (!controller.signal.aborted) setStatus('Error checking model: ' + errorMessage(error)) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    get('/api/rag/status', undefined, { signal: controller.signal }).then(result => { setRagLoaded(result.loaded ?? ''); setRagPath(result.loaded ?? '') }).catch(() => {})
    get('/api/facts/status', undefined, { signal: controller.signal }).then(result => { setFactsLoaded(result.loaded ?? ''); setFactsPath(result.loaded ?? '') }).catch(() => {})
    return () => { controller.abort(); streamAbort.current?.abort() }
  }, [])

  async function loadRag() {
    const path = ragPath.trim()
    if (!path || ragBusy || busy) return
    setRagBusy(true)
    try { await post('/api/rag/load', { path }); setRagLoaded(path); setShowRag(false); setStatus('') }
    catch (error) { setStatus('RAG error: ' + errorMessage(error)) }
    finally { setRagBusy(false) }
  }
  async function unloadRag() {
    setRagBusy(true)
    try { await post('/api/rag/unload'); setRagLoaded(''); setShowRag(false) }
    catch (error) { setStatus('RAG error: ' + errorMessage(error)) }
    finally { setRagBusy(false) }
  }
  async function loadFacts() {
    const path = factsPath.trim()
    if (!path || factsBusy || busy) return
    setFactsBusy(true)
    try { await post('/api/facts/load', { path }); setFactsLoaded(path); setShowFacts(false); setStatus('') }
    catch (error) { setStatus('Facts error: ' + errorMessage(error)) }
    finally { setFactsBusy(false) }
  }
  async function unloadFacts() {
    setFactsBusy(true)
    try { await post('/api/facts/unload'); setFactsLoaded(''); setShowFacts(false) }
    catch (error) { setStatus('Facts error: ' + errorMessage(error)) }
    finally { setFactsBusy(false) }
  }

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  function resizeTextarea() {
    const el = taRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 180) + 'px'
  }

  async function loadModel(path: string) {
    if (!path || busy || loading) return
    setLoading(true); setStatus('Loading…')
    try { await post('/api/chat/load', { path }); setLoaded(path); localStorage.setItem(STORAGE_KEY, path); setStatus(''); setMsgs([]) }
    catch (error) { setStatus('Error: ' + errorMessage(error)) }
    finally { setLoading(false); setShowPicker(false) }
  }

  async function send() {
    if (!input.trim() || busy || !loaded || loading) return
    const history: Msg[] = [...msgs.filter(message => !message.error), { role: 'user', content: input.trim() }]
    const assistantIndex = history.length
    setMsgs([...history, { role: 'assistant', content: '' }]); setInput(''); setBusy(true); setStatus('')
    const controller = new AbortController(); streamAbort.current = controller
    if (taRef.current) taRef.current.style.height = 'auto'
    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), signal: controller.signal,
        body: JSON.stringify({ messages: history.map(message => [message.role, message.content]), max_new_tokens: maxTokens, temperature, top_k: topK }),
      })
      await consumeSSE(response, message => {
        if (message.type === 'warning') { setStatus('Warning: ' + message.text); return }
        if (message.type !== 'chunk') return
        setMsgs(current => current.map((item, index) => index === assistantIndex ? { ...item, content: item.content + message.text, source: message.source ?? item.source } : item))
      }, controller.signal)
    } catch (error) {
      const detail = controller.signal.aborted ? 'Generation cancelled' : 'Error: ' + errorMessage(error)
      setStatus(detail)
      setMsgs(current => current.map((item, index) => index === assistantIndex ? { ...item, content: item.content || detail, error: true } : item))
    } finally { streamAbort.current = null; setBusy(false) }
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
                {m.source && <div className="chip" style={{ marginBottom: 6 }}>Source: {m.source}</div>}
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
              aria-label="Message"
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send() } }}
              placeholder={loaded ? 'Message…' : 'Select a model below, then message…'}
              disabled={!loaded || busy || loading} rows={1}
              style={{
                border: 'none', outline: 'none', background: 'transparent', resize: 'none',
                color: 'var(--color-ink)', fontFamily: 'var(--font-sans)', fontSize: 15,
                lineHeight: 1.6, width: '100%', maxHeight: 180, overflowY: 'auto',
              }}
            />

            {/* Bottom bar: model picker + RAG pill + send */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              {/* Model pill — Claude.ai style */}
              <div style={{ position: 'relative' }} onClick={e => e.stopPropagation()}>
                <button aria-label="Select model" aria-expanded={showPicker} disabled={loading || busy} onClick={() => setShowPicker(s => !s)} style={{
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
                        <button key={m.path} disabled={loading || busy} onClick={() => loadModel(m.path)} style={{
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

              {/* RAG index pill */}
              <div style={{ position: 'relative' }} onClick={e => e.stopPropagation()}>
                <button onClick={() => setShowRag(s => !s)} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '5px 10px', borderRadius: 20,
                  border: '1px solid var(--color-line-2)',
                  background: showRag ? 'var(--color-surface-3)' : 'transparent',
                  color: ragLoaded ? 'var(--color-ink-2)' : 'var(--color-ink-3)',
                  cursor: 'pointer', fontSize: 12, fontFamily: 'var(--font-mono)',
                  maxWidth: 200, overflow: 'hidden',
                }}>
                  {ragBusy
                    ? <Loader2 size={11} className="spin" />
                    : <Database size={11} style={{ flexShrink: 0, color: ragLoaded ? 'var(--color-pos)' : 'var(--color-ink-3)' }} />
                  }
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {ragLoaded ? ragLoaded.split(/[\\/]/).pop() : 'No index'}
                  </span>
                </button>

                {showRag && (
                  <div style={{
                    position: 'absolute', bottom: 'calc(100% + 8px)', left: 0, zIndex: 50,
                    background: 'var(--color-surface-2)', border: '1px solid var(--color-line-2)',
                    borderRadius: 12, padding: 10, minWidth: 300,
                    boxShadow: '0 -8px 32px rgba(0,0,0,0.4)',
                    display: 'flex', flexDirection: 'column', gap: 8,
                  }}>
                    <div style={{ fontSize: 11, color: 'var(--color-ink-3)', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase' }}>RAG Index</div>
                    <input
                      value={ragPath}
                      onChange={e => setRagPath(e.target.value)}
                      placeholder="Path to index directory…"
                      style={{
                        border: '1px solid var(--color-line-2)', borderRadius: 8, padding: '6px 10px',
                        background: 'var(--color-surface-3)', color: 'var(--color-ink)',
                        fontFamily: 'var(--font-mono)', fontSize: 12, outline: 'none', width: '100%',
                      }}
                    />
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button onClick={loadRag} disabled={ragBusy || busy || !ragPath.trim()} style={{
                        flex: 1, padding: '6px 0', borderRadius: 8, border: 'none', cursor: 'pointer',
                        background: 'var(--color-accent)', color: 'var(--color-accent-ink)', fontSize: 12,
                      }}>Load</button>
                      {ragLoaded && (
                        <button disabled={ragBusy || busy} onClick={unloadRag} style={{
                          padding: '6px 12px', borderRadius: 8, border: '1px solid var(--color-line-2)',
                          background: 'transparent', color: 'var(--color-ink-3)', cursor: 'pointer', fontSize: 12,
                        }}>Unload</button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Facts index pill — answers personal questions from a facts KB,
                  bypassing the model on a confident retrieval hit */}
              <div style={{ position: 'relative' }} onClick={e => e.stopPropagation()}>
                <button onClick={() => setShowFacts(s => !s)} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '5px 10px', borderRadius: 20,
                  border: '1px solid var(--color-line-2)',
                  background: showFacts ? 'var(--color-surface-3)' : 'transparent',
                  color: factsLoaded ? 'var(--color-ink-2)' : 'var(--color-ink-3)',
                  cursor: 'pointer', fontSize: 12, fontFamily: 'var(--font-mono)',
                  maxWidth: 200, overflow: 'hidden',
                }}>
                  {factsBusy
                    ? <Loader2 size={11} className="spin" />
                    : <Sparkles size={11} style={{ flexShrink: 0, color: factsLoaded ? 'var(--color-pos)' : 'var(--color-ink-3)' }} />
                  }
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {factsLoaded ? factsLoaded.split(/[\\/]/).pop() : 'No facts'}
                  </span>
                </button>

                {showFacts && (
                  <div style={{
                    position: 'absolute', bottom: 'calc(100% + 8px)', left: 0, zIndex: 50,
                    background: 'var(--color-surface-2)', border: '1px solid var(--color-line-2)',
                    borderRadius: 12, padding: 10, minWidth: 300,
                    boxShadow: '0 -8px 32px rgba(0,0,0,0.4)',
                    display: 'flex', flexDirection: 'column', gap: 8,
                  }}>
                    <div style={{ fontSize: 11, color: 'var(--color-ink-3)', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase' }}>Facts KB</div>
                    <div style={{ fontSize: 11, color: 'var(--color-ink-3)', lineHeight: 1.4 }}>
                      Personal questions are answered directly from this index, bypassing the model.
                    </div>
                    <input
                      value={factsPath}
                      onChange={e => setFactsPath(e.target.value)}
                      placeholder="Path to facts index directory…"
                      style={{
                        border: '1px solid var(--color-line-2)', borderRadius: 8, padding: '6px 10px',
                        background: 'var(--color-surface-3)', color: 'var(--color-ink)',
                        fontFamily: 'var(--font-mono)', fontSize: 12, outline: 'none', width: '100%',
                      }}
                    />
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button onClick={loadFacts} disabled={factsBusy || busy || !factsPath.trim()} style={{
                        flex: 1, padding: '6px 0', borderRadius: 8, border: 'none', cursor: 'pointer',
                        background: 'var(--color-accent)', color: 'var(--color-accent-ink)', fontSize: 12,
                      }}>Load</button>
                      {factsLoaded && (
                        <button disabled={factsBusy || busy} onClick={unloadFacts} style={{
                          padding: '6px 12px', borderRadius: 8, border: '1px solid var(--color-line-2)',
                          background: 'transparent', color: 'var(--color-ink-3)', cursor: 'pointer', fontSize: 12,
                        }}>Unload</button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Generation params pill */}
              <div style={{ position: 'relative' }} onClick={e => e.stopPropagation()}>
                <button onClick={() => setShowParams(s => !s)} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '5px 10px', borderRadius: 20,
                  border: '1px solid var(--color-line-2)',
                  background: showParams ? 'var(--color-surface-3)' : 'transparent',
                  color: 'var(--color-ink-2)',
                  cursor: 'pointer', fontSize: 12, fontFamily: 'var(--font-mono)',
                }}>
                  <Sliders size={11} style={{ flexShrink: 0 }} />
                  <span>temp {temperature.toFixed(2)}</span>
                </button>

                {showParams && (
                  <div style={{
                    position: 'absolute', bottom: 'calc(100% + 8px)', left: 0, zIndex: 50,
                    background: 'var(--color-surface-2)', border: '1px solid var(--color-line-2)',
                    borderRadius: 12, padding: 14, minWidth: 300,
                    boxShadow: '0 -8px 32px rgba(0,0,0,0.4)',
                    display: 'flex', flexDirection: 'column', gap: 14,
                  }}>
                    <div style={{ fontSize: 11, color: 'var(--color-ink-3)', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase' }}>Generation</div>
                    <Slider label="Temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} />
                    <Slider label="Top-K" value={topK} min={0} max={200} step={1} onChange={v => setTopK(Math.round(v))} />
                    <Slider label="Max new tokens" value={maxTokens} min={16} max={4096} step={16} onChange={v => setMaxTokens(Math.round(v))} />
                  </div>
                )}
              </div>
              </div>

              {/* Send button */}
              <button aria-label={busy ? 'Stop generation' : 'Send message'} onClick={() => busy ? streamAbort.current?.abort() : send()} disabled={!loaded || loading || (!busy && !input.trim())} style={{
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

          {status && <div role="status" aria-live="polite" style={{ fontSize: 12, color: 'var(--color-neg)', marginTop: 6, paddingLeft: 4 }}>{status}</div>}
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
