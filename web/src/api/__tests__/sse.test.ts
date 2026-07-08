import { describe, it, expect } from 'vitest'
import { parseSSEChunk } from '../sse'

describe('parseSSEChunk', () => {
  it('parses complete JSON event frames', () => {
    const buf = 'data: {"type":"token","text":"hi"}\n\n'
    const { events, rest } = parseSSEChunk(buf)
    expect(events).toEqual([{ type: 'token', text: 'hi' }])
    expect(rest).toBe('')
  })

  it('parses multiple frames and keeps the trailing partial', () => {
    const buf = 'data: {"type":"token","text":"a"}\n\ndata: {"type":"token","text":"b"}'
    const { events, rest } = parseSSEChunk(buf)
    expect(events).toEqual([{ type: 'token', text: 'a' }])
    expect(rest).toBe('data: {"type":"token","text":"b"}')
  })

  it('treats [DONE] sentinel as a done event', () => {
    const { events } = parseSSEChunk('data: [DONE]\n\n')
    expect(events).toEqual([{ type: 'done' }])
  })

  it('ignores malformed JSON frames', () => {
    const { events } = parseSSEChunk('data: not-json\n\n')
    expect(events).toEqual([])
  })

  it('parses sources + trace events', () => {
    const buf = 'data: {"type":"sources","items":[{"source":"HR/x.md","page":1,"department":"HR","score":0.9,"preview":"p"}]}\n\n'
    const { events } = parseSSEChunk(buf)
    expect(events[0].type).toBe('sources')
    if (events[0].type === 'sources') expect(events[0].items[0].source).toBe('HR/x.md')
  })
})
