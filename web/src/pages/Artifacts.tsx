import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { get, errorMessage } from '../lib/api'
import { Card, CardTitle, PageHeader, Field, Button, Empty, StatusText } from '../lib/ui'
export default function Artifacts() {
  const [filter, setFilter] = useState(''), [selected, setSelected] = useState('')
  const query = useQuery({ queryKey: ['artifacts'], queryFn: () => get('/api/artifacts') })
  const entries: any[] = (query.data?.artifacts ?? []).filter((entry: any) => (entry.path + ' ' + JSON.stringify(entry.manifest)).toLowerCase().includes(filter.toLowerCase()))
  const active = entries.find(entry => entry.path === selected)
  return <div style={{ padding: 28 }}><PageHeader title="Artifacts" subtitle="Saved model, data, run and evaluation provenance." right={<Button onClick={() => query.refetch()}>Refresh artifacts</Button>} />
    {query.error && <StatusText msg={'Error: ' + errorMessage(query.error)} />}
    <Field label="Filter artifacts"><input className="input" value={filter} onChange={event => setFilter(event.target.value)} /></Field>
    <div className="studio-split" style={{ display: 'grid', gridTemplateColumns: 'minmax(300px,1fr) minmax(280px,1fr)', gap: 14, marginTop: 14 }}><Card><CardTitle>Artifact catalog</CardTitle>{!entries.length && <Empty>No matching manifests.</Empty>}
      {entries.map(entry => <button className="btn" type="button" key={entry.path} aria-pressed={selected === entry.path} onClick={() => setSelected(entry.path)} style={{ width: '100%', marginBottom: 8, flexDirection: 'column', alignItems: 'flex-start' }}><strong>{entry.manifest.kind ?? entry.manifest.artifact_type ?? 'Artifact'}</strong><span>{entry.path}</span><small>{entry.manifest.artifact_id ?? entry.manifest.id}</small></button>)}
    </Card><Card><CardTitle>Provenance</CardTitle>{active ? <><dl><dt>Schema version</dt><dd>{active.manifest.schema_version}</dd><dt>Owner</dt><dd>{active.manifest.owner ?? active.manifest.metadata?.owner ?? 'Not recorded'}</dd><dt>Retention</dt><dd>{active.manifest.retention ?? active.manifest.metadata?.retention ?? 'Not recorded'}</dd><dt>Content fingerprint</dt><dd style={{ overflowWrap: 'anywhere' }}>{active.manifest.content_fingerprint ?? active.manifest.fingerprint ?? 'Not recorded'}</dd></dl><pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{JSON.stringify(active.manifest, null, 2)}</pre></> : <Empty>Select a manifest to inspect its recorded identity, parents and provenance.</Empty>}</Card></div>
  </div>
}
