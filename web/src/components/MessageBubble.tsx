import type { ChatMessage } from '../hooks/useSSEChat'
import { SourceCard } from './SourceCard'
import { TracePanel } from './TracePanel'

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-lg px-3 py-2 ${isUser ? 'bg-blue-600 text-white' : 'bg-white border border-slate-200 text-slate-800'}`}>
        <div className="whitespace-pre-wrap break-words">{msg.content || (msg.error ? '' : '…')}</div>
        {msg.error && <div className="mt-1 text-red-600 text-sm">⚠ {msg.error}</div>}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="mt-2 space-y-1">
            <div className="text-xs text-slate-500 font-medium">引用来源</div>
            {msg.sources.map((s, i) => <SourceCard key={i} s={s} index={i + 1} />)}
          </div>
        )}
        {!isUser && msg.trace && <TracePanel trace={msg.trace} />}
      </div>
    </div>
  )
}
