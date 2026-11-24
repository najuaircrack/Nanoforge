type RequestOptions = { signal?: AbortSignal }
const TOKEN_KEY = 'latticekin_api_token'
export function getAuthToken() { return sessionStorage.getItem(TOKEN_KEY) ?? '' }
export function setAuthToken(token: string) { token ? sessionStorage.setItem(TOKEN_KEY, token.trim()) : sessionStorage.removeItem(TOKEN_KEY) }
export function authHeaders(extra: Record<string, string> = {}) { const token = getAuthToken(); return { ...extra, ...(token ? { Authorization: `Bearer ${token}` } : {}) } }
export function errorMessage(error: unknown) { return error instanceof Error ? error.message : String(error) }
async function responseError(res: Response) {
  const text = await res.text()
  try { const data = JSON.parse(text); return new Error(typeof data.detail === 'string' ? data.detail : data.error ?? JSON.stringify(data.detail ?? data)) }
  catch { return new Error(text || `HTTP ${res.status}`) }
}
async function request(url: string, init: RequestInit = {}) {
  const res = await fetch(url, { ...init, headers: authHeaders(init.headers as Record<string, string>) })
  if (!res.ok) throw await responseError(res)
  const data = await res.json()
  if (data?.ok === false) throw new Error(data.error ?? 'Request failed')
  return data
}
export const get = (url: string, params?: Record<string, any>, options?: RequestOptions) => {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params ?? {})) if (value != null && value !== '') query.set(key, String(value))
  return request(query.size ? `${url}?${query}` : url, { signal: options?.signal })
}
export const post = (url: string, body?: any, options?: RequestOptions) => request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}), signal: options?.signal })
export async function getText(url: string, params?: Record<string, string>) {
  const query = new URLSearchParams(params)
  const res = await fetch(query.size ? `${url}?${query}` : url, { headers: authHeaders() })
  if (!res.ok) throw await responseError(res)
  return res.text()
}
export async function upload(file: File, subdir = 'uploads') {
  const data = new FormData(); data.append('file', file); data.append('subdir', subdir)
  return request('/api/data/upload', { method: 'POST', body: data })
}
export async function consumeSSE(res: Response, onEvent: (message: any) => void, signal?: AbortSignal) {
  if (!res.ok) throw await responseError(res)
  if (!res.headers.get('content-type')?.includes('text/event-stream') || !res.body) throw new Error('Expected an event stream')
  const reader = res.body.getReader(), decoder = new TextDecoder()
  let buffer = '', completed = false
  try {
    while (!completed) {
      if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError')
      const { done, value } = await reader.read()
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
      const frames = buffer.replace(/\r\n/g, '\n').split('\n\n'); buffer = frames.pop() ?? ''
      for (const frame of frames) {
        const payload = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n')
        if (!payload) continue
        const message = JSON.parse(payload)
        if (message.type === 'error') throw new Error(message.text ?? message.error ?? 'Stream failed')
        onEvent(message); if (message.type === 'done') completed = true
      }
      if (done) break
    }
    if (!completed) throw new Error('Stream interrupted before completion')
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock() }
}
export function fmtBytes(n: number) { if (n == null) return '—'; const units = ['B', 'KB', 'MB', 'GB', 'TB']; let index = 0; while (n >= 1024 && index < units.length - 1) { n /= 1024; index++ }; return `${n.toFixed(index ? 1 : 0)} ${units[index]}` }
