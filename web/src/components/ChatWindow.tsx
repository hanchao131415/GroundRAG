import { useState } from 'react'
import { MessageBubble } from './MessageBubble'
import type { ChatMessage } from '../hooks/useSSEChat'
import { useLang } from '../i18n'

export function ChatWindow({
  messages, asking, onAsk, onStop,
}: {
  messages: ChatMessage[]
  asking: boolean
  onAsk: (q: string) => void
  onStop: () => void
}) {
  const { t } = useLang()
  const [input, setInput] = useState('')
  const submit = () => {
    const q = input.trim()
    if (!q || asking) return
    onAsk(q)
    setInput('')
  }
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto space-y-3 p-3">
        {messages.length === 0 && (
          <div className="text-slate-400 text-sm p-4 text-center">
            {t('chat_empty')}
          </div>
        )}
        {messages.map((m) => <MessageBubble key={m.id} msg={m} />)}
      </div>
      <div className="border-t border-slate-200 p-2 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          placeholder={t('input_placeholder')}
          className="flex-1 border border-slate-300 rounded px-3 py-2 outline-none focus:border-blue-500"
        />
        <button
          onClick={asking ? onStop : submit}
          disabled={!asking && !input.trim()}
          className="px-4 py-2 rounded bg-blue-600 text-white disabled:bg-slate-300"
        >{asking ? t('stop') : t('send')}</button>
      </div>
    </div>
  )
}
