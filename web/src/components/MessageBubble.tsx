import type { ChatMessage } from '../hooks/useSSEChat'
import { SourceCard } from './SourceCard'
import { TracePanel } from './TracePanel'
import { useLang } from '../i18n'

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  const { t } = useLang()
  const isUser = msg.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-lg px-3 py-2 ${isUser ? 'bg-blue-600 text-white' : 'bg-white border border-slate-200 text-slate-800'}`}>
        <div className="whitespace-pre-wrap break-words">{msg.content || (msg.error ? '' : t('assistant_loading'))}</div>
        {msg.error && <div className="mt-1 text-red-600 text-sm">{t('error_prefix')} {msg.error}</div>}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="mt-2 space-y-1">
            <div className="text-xs text-slate-500 font-medium">{t('sources_label')}</div>
            {msg.sources.map((s, i) => <SourceCard key={i} s={s} index={i + 1} />)}
          </div>
        )}
        {!isUser && msg.trace && <TracePanel trace={msg.trace} />}
      </div>
    </div>
  )
}
