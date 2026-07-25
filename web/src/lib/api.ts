import axios from 'axios'

export const get = (url: string, params?: any) => axios.get(url, { params }).then(r => r.data)
export const post = (url: string, body?: any) => axios.post(url, body).then(r => r.data)

export function fmtBytes(n: number): string {
  if (n == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++ }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`
}

// Stream a job's log lines over SSE. Returns an unsubscribe fn.
export function streamJob(jobId: string, onLine: (line: string) => void, onDone: (code: number) => void): () => void {
  const es = new EventSource(`/stream/job/${jobId}`)
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data)
    if (msg.type === 'line') onLine(msg.line)
    else if (msg.type === 'done') { onDone(msg.code); es.close() }
  }
  es.onerror = () => es.close()
  return () => es.close()
}
