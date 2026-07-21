import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getDemoUsers } from '../../api/client'
import { useAuth } from '../useAuth'

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>()
  return { ...actual, getDemoUsers: vi.fn(), login: vi.fn() }
})

describe('useAuth readiness retry', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
  })

  afterEach(() => vi.useRealTimers())

  it('retries demo users after a temporary 503', async () => {
    vi.mocked(getDemoUsers)
      .mockRejectedValueOnce(new ApiError(503, 'RAG_NOT_READY', 'initializing'))
      .mockResolvedValueOnce([{ user_id: 'zhangsan', name: '张三', departments: ['HR'], role: '员工' }])

    const { result } = renderHook(() => useAuth())
    await act(async () => {
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1000)
      await Promise.resolve()
    })

    expect(result.current.demoUsers).toHaveLength(1)
    expect(getDemoUsers).toHaveBeenCalledTimes(2)
  })
})
