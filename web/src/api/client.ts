import type { DemoUser, Source, SSEEvent, Trace } from '../types'
import { parseSSEChunk } from './sse'

let _token: string | null = localStorage.getItem('groundrag_token')
export function setToken(t: string | null) {
  _token = t
  if (t) localStorage.setItem('groundrag_token', t)
  else localStorage.removeItem('groundrag_token')
}
function authHeaders(): Record<string, string> {
  return _token ? { Authorization: `Bearer ${_token}` } : {}
}

export async function login(user_id: string): Promise<string> {
  const r = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id }),
  })
  if (!r.ok) throw new Error(`登录失败: ${r.status}`)
  const data = await r.json()
  setToken(data.access_token)
  return data.access_token
}

export async function getDemoUsers(): Promise<DemoUser[]> {
  const r = await fetch('/auth/demo-users')
  if (!r.ok) return []
  const data = await r.json()
  return data.users
}

export interface SearchResult { source: string; chunk_index: number | null; score: number; preview: string }

export async function search(question: string, top_k = 3): Promise<SearchResult[]> {
  const r = await fetch('/api/v1/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ question, top_k }),
  })
  if (!r.ok) throw new Error(`检索失败: ${r.status}`)
  const data = await r.json()
  return data.results
}

export async function getStats(): Promise<Record<string, unknown>> {
  const r = await fetch('/api/v1/stats', { headers: authHeaders() })
  if (!r.ok) throw new Error(`统计失败: ${r.status}`)
  return r.json()
}

/**
 * 流式问答：POST /api/v1/chat (stream)，读 ReadableStream，按帧解析，回调每个事件。
 */
export async function streamChat(
  question: string,
  onEvent: (e: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ question, stream: true }),
    signal,
  })
  if (!r.ok || !r.body) throw new Error(`问答失败: ${r.status}`)

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const { events, rest } = parseSSEChunk(buffer)
    buffer = rest
    for (const e of events) onEvent(e)
  }
}
