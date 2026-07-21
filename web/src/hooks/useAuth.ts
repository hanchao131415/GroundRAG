import { useEffect, useState, useCallback } from 'react'
import type { DemoUser } from '../types'
import { ApiError, login, getDemoUsers } from '../api/client'

export function useAuth() {
  const [demoUsers, setDemoUsers] = useState<DemoUser[]>([])
  const [currentUserId, setCurrentUserId] = useState<string | null>(null)

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined
    let disposed = false
    const load = () => {
      getDemoUsers().then((users) => {
        if (!disposed) setDemoUsers(users)
      }).catch((err) => {
        if (!disposed && err instanceof ApiError && err.status === 503) {
          timer = setTimeout(load, 1000)
        }
      })
    }
    load()
    return () => {
      disposed = true
      if (timer) clearTimeout(timer)
    }
  }, [])

  const loginAs = useCallback(async (userId: string | null) => {
    if (!userId) {
      setCurrentUserId(null)
      return
    }
    await login(userId)
    setCurrentUserId(userId)
  }, [])

  const currentUser = demoUsers.find((u) => u.user_id === currentUserId) ?? null
  return { demoUsers, currentUser, loginAs }
}
