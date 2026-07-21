import { useState, useCallback, useRef } from 'react'
import type { Source, Trace } from '../types'
import { streamChat } from '../api/client'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  trace?: Trace
  error?: string
}

let _seq = 0
const nextId = () => `m${++_seq}`

export function useSSEChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [asking, setAsking] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const ask = useCallback(async (question: string) => {
    const userMsg: ChatMessage = { id: nextId(), role: 'user', content: question }
    const assistantId = nextId()
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '' }
    setMessages((m) => [...m, userMsg, assistantMsg])
    setAsking(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl

    const patch = (p: Partial<ChatMessage>) =>
      setMessages((m) => m.map((x) => (x.id === assistantId ? { ...x, ...p } : x)))
    const append = (text: string) =>
      setMessages((m) => m.map((x) => (x.id === assistantId ? { ...x, content: x.content + text } : x)))

    try {
      await streamChat(
        question,
        (e) => {
          if (e.type === 'sources') patch({ sources: e.items })
          else if (e.type === 'token') append(e.text)
          else if (e.type === 'trace') patch({ trace: e.trace })
          else if (e.type === 'error') patch({ error: e.message })
        },
        ctrl.signal,
      )
    } catch (err) {
      if ((err as Error).name !== 'AbortError') patch({ error: (err as Error).message })
    } finally {
      setAsking(false)
    }
  }, [])

  const stop = useCallback(() => abortRef.current?.abort(), [])

  const clear = useCallback(() => setMessages([]), [])
  return { messages, ask, asking, stop, clear }
}
