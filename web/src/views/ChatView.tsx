import { useEffect } from 'react'
import { ChatWindow } from '../components/ChatWindow'
import { UserSwitcher } from '../components/UserSwitcher'
import { useAuth } from '../hooks/useAuth'
import { useSSEChat } from '../hooks/useSSEChat'
import { useLang } from '../i18n'

export function ChatView() {
  const { t } = useLang()
  const { demoUsers, currentUser, loginAs } = useAuth()
  const { messages, ask, asking, stop, clear } = useSSEChat()

  // 切换用户时清空对话，避免跨用户串内容
  useEffect(() => { clear() }, [currentUser, clear])

  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-200 bg-slate-50">
        <UserSwitcher users={demoUsers} current={currentUser} onSelect={loginAs} />
        <span className="text-xs text-slate-400">
          {currentUser ? `${t('visible_depts')} ${currentUser.departments.join(', ')}` : t('no_filter')}
        </span>
      </div>
      <ChatWindow messages={messages} asking={asking} onAsk={ask} onStop={stop} />
    </div>
  )
}
