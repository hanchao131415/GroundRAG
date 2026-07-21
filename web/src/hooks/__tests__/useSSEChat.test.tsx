import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useSSEChat } from '../useSSEChat'
import { streamChat } from '../../api/client'

vi.mock('../../api/client', () => ({ streamChat: vi.fn() }))

describe('useSSEChat stop', () => {
  beforeEach(() => vi.clearAllMocks())

  it('aborts the active stream without showing an error', async () => {
    vi.mocked(streamChat).mockImplementation((_question, _onEvent, signal) => new Promise((_resolve, reject) => {
      signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
    }))
    const { result } = renderHook(() => useSSEChat())

    act(() => { void result.current.ask('hello') })
    await waitFor(() => expect(result.current.asking).toBe(true))
    act(() => result.current.stop())
    await waitFor(() => expect(result.current.asking).toBe(false))

    expect(result.current.messages.at(-1)?.error).toBeUndefined()
  })
})
