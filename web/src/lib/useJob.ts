import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { errorMessage, get, post } from './api'
export type Job = { job_id?: string; done: boolean; code: number | null; lines: string[]; run_dir?: string; type?: string }
export function useJob(scope: string) {
  const key = 'latticekin_job_' + scope
  const [jobId, setJobId] = useState<string | null>(() => sessionStorage.getItem(key))
  const [pending, setPending] = useState(false), [actionError, setActionError] = useState('')
  const client = useQueryClient()
  const query = useQuery<Job>({ queryKey: ['job', jobId], enabled: !!jobId, staleTime: 0,
    queryFn: ({ signal }) => get('/api/jobs/' + encodeURIComponent(jobId!), undefined, { signal }),
    refetchInterval: state => state.state.data?.done ? false : 1000, retry: 1,
  })
  function select(id: string | null) { setJobId(id); id ? sessionStorage.setItem(key, id) : sessionStorage.removeItem(key); setActionError('') }
  async function start(body: Record<string, any>) {
    setPending(true); setActionError('')
    try {
      const result = await post('/api/jobs/start', body)
      if (typeof result.job_id !== 'string' || !result.job_id) throw new Error('Server returned no job identity')
      client.setQueryData(['job', result.job_id], { done: false, code: null, lines: [], run_dir: result.run_dir, type: body.type })
      select(result.job_id); await client.invalidateQueries({ queryKey: ['jobs'] })
      return result
    } catch (error) { setActionError(errorMessage(error)); throw error }
    finally { setPending(false) }
  }
  async function stop() {
    if (!jobId) return
    setPending(true); setActionError('')
    try { await post('/api/jobs/stop', { job_id: jobId }); await query.refetch() }
    catch (error) { setActionError(errorMessage(error)) } finally { setPending(false) }
  }
  return { jobId, select, start, stop, pending, job: query.data, running: pending || (!!jobId && !query.data?.done), error: actionError || (query.error ? 'Cannot confirm job status: ' + errorMessage(query.error) : ''), refetch: query.refetch }
}
