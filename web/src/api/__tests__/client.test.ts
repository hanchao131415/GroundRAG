import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getDemoUsers, getStats, setToken, streamChat } from '../client'

describe('API errors', () => {
  beforeEach(() => {
    localStorage.clear()
    setToken(null)
    vi.restoreAllMocks()
  })

  it('preserves backend status and structured detail', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: { code: 'RAG_NOT_READY', status: 'initializing', message: 'RAG is initializing' },
    }), { status: 503, headers: { 'Content-Type': 'application/json' } })))

    await expect(streamChat('q', () => undefined)).rejects.toMatchObject({
      status: 503,
      code: 'RAG_NOT_READY',
      message: 'RAG is initializing',
    })
  })

  it('clears the stored token after a 401', async () => {
    setToken('expired')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'expired' }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    })))

    await expect(getStats()).rejects.toMatchObject({ status: 401 })
    expect(localStorage.getItem('groundrag_token')).toBeNull()
  })

  it('does not hide readiness errors while loading demo users', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: { code: 'RAG_NOT_READY', status: 'initializing', message: 'initializing' },
    }), { status: 503, headers: { 'Content-Type': 'application/json' } })))

    await expect(getDemoUsers()).rejects.toMatchObject({ status: 503, code: 'RAG_NOT_READY' })
  })
})
