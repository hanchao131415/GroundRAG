import type { DemoUser, SSEEvent } from '../types'
import { parseSSEChunk } from './sse'

let _token: string | null = localStorage.getItem('groundrag_token')

export class ApiError extends Error {
  readonly status: number
  readonly code: string | undefined

  constructor(
    status: number,
    code: string | undefined,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

async function throwApiError(response: Response): Promise<never> {
  let detail: unknown
  try {
    detail = (await response.json()).detail
  } catch {
    detail = undefined
  }
  if (response.status === 401) setToken(null)
  if (detail && typeof detail === 'object') {
    const value = detail as { code?: string; message?: string }
    throw new ApiError(response.status, value.code, value.message || `请求失败: ${response.status}`)
  }
  throw new ApiError(response.status, undefined, typeof detail === 'string' ? detail : `请求失败: ${response.status}`)
}
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
  if (!r.ok) await throwApiError(r)
  const data = await r.json()
  setToken(data.access_token)
  return data.access_token
}

export async function getDemoUsers(): Promise<DemoUser[]> {
  const r = await fetch('/auth/demo-users')
  if (!r.ok) await throwApiError(r)
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
  if (!r.ok) await throwApiError(r)
  const data = await r.json()
  return data.results
}

export async function getStats(): Promise<Record<string, unknown>> {
  const r = await fetch('/api/v1/stats', { headers: authHeaders() })
  if (!r.ok) await throwApiError(r)
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
  if (!r.ok) await throwApiError(r)
  if (!r.body) throw new ApiError(r.status, undefined, '问答响应为空')

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
