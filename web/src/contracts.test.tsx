import { describe, it, expect, vi, afterEach } from 'vitest'
import { cleanup, render, screen, fireEvent } from '@testing-library/react'
import { get, getText, consumeSSE } from './lib/api'
import { ConfigField } from './lib/ConfigField'
import { parseField } from './lib/contracts'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('configuration shape contracts', () => {
  it('preserves tuples/arrays and dictionaries and rejects scalar corruption', () => {
    expect(parseField({ name: 'betas', type: 'tuple[float, float]', widget: 'number', default: [0.9, 0.95] }, '[0.8,0.9]')).toEqual([0.8, 0.9])
    expect(() => parseField({ name: 'betas', type: 'tuple[float, float]', widget: 'number' }, '0.8')).toThrow(/array/i)
    expect(parseField({ name: 'sources', type: 'list[dict[str, Any]] | None', widget: 'text' }, '[{"path":"synthetic.bin"}]')).toEqual([{ path: 'synthetic.bin' }])
    expect(parseField({ name: 'dataset_weights', type: 'dict[str, float] | None', widget: 'number' }, '{"one":1}')).toEqual({ one: 1 })
  })
  it('allows nullable booleans to return to automatic', () => {
    const change = vi.fn()
    render(<ConfigField field={{ name: 'is_causal', type: 'bool | None', widget: 'toggle' }} value={true} onChange={change} />)
    fireEvent.change(screen.getByRole('combobox', { name: 'Is Causal' }), { target: { value: 'null' } })
    expect(change).toHaveBeenCalledWith(null)
  })
  it('keeps invalid JSON visible and never commits a broken structured value', async () => {
    const change = vi.fn()
    render(<ConfigField field={{ name: 'stop_tokens', type: 'list[str]', widget: 'text' }} value={['END']} onChange={change} />)
    fireEvent.change(screen.getByRole('textbox', { name: 'Stop Tokens' }), { target: { value: '[' } })
    expect(change).not.toHaveBeenCalled()
    expect(screen.getByRole('alert').textContent).toMatch(/JSON/i)
    fireEvent.change(screen.getByRole('textbox', { name: 'Stop Tokens' }), { target: { value: '["STOP"]' } })
    expect(change).toHaveBeenCalledWith(['STOP'])
  })
})

describe('request and streaming failure paths', () => {
  it('surfaces HTTP failure and API failure instead of leaving blank output', async () => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => Promise.resolve(new Response('{"detail":"synthetic failure"}', { status: 500, headers: { 'Content-Type': 'application/json' } }))))
    await expect(get('/synthetic')).rejects.toThrow('synthetic failure')
    await expect(getText('/synthetic')).rejects.toThrow('synthetic failure')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"ok":false,"error":"denied"}', { headers: { 'Content-Type': 'application/json' } })))
    await expect(get('/synthetic')).rejects.toThrow('denied')
  })
  it('decodes split UTF-8 chunks and requires a successful terminal event', async () => {
    const bytes = new TextEncoder().encode('data: {"type":"chunk","text":"hé"}\n\ndata: {"type":"done"}\n\n')
    const stream = new ReadableStream({ start(c) { c.enqueue(bytes.slice(0, 34)); c.enqueue(bytes.slice(34)); c.close() } })
    const seen: string[] = []
    await consumeSSE(new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } }), m => { if (m.type === 'chunk') seen.push(m.text) })
    expect(seen).toEqual(['hé'])
    await expect(consumeSSE(new Response('data: {"type":"chunk","text":"partial"}\n\n', { headers: { 'Content-Type': 'text/event-stream' } }), () => {})).rejects.toThrow(/interrupted/i)
  })
  it('surfaces plain JSON errors before parsing an SSE response', async () => {
    await expect(consumeSSE(new Response('{"detail":"bad request"}', { status: 422, headers: { 'Content-Type': 'application/json' } }), () => {})).rejects.toThrow('bad request')
  })
})

