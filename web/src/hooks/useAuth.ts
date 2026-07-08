import { useEffect, useState, useCallback } from 'react'
import type { DemoUser } from '../types'
import { login, getDemoUsers } from '../api/client'

export function useAuth() {
  const [demoUsers, setDemoUsers] = useState<DemoUser[]>([])
  const [currentUserId, setCurrentUserId] = useState<string | null>(null)

  useEffect(() => {
    getDemoUsers().then(setDemoUsers).catch(() => {})
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
