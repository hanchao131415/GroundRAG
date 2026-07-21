import { useState } from 'react'
import { ChatView } from './views/ChatView'
import { SearchView } from './views/SearchView'
import { StatsView } from './views/StatsView'
import { KnowledgeBaseView } from './views/KnowledgeBaseView'
import { LangProvider, useLang, LanguageToggle } from './i18n'
import { useAuth } from './hooks/useAuth'

type View = 'chat' | 'search' | 'stats' | 'knowledge'

function Shell() {
  const { t } = useLang()
  const { demoUsers, currentUser, loginAs } = useAuth()
  const [view, setView] = useState<View>('chat')
  const tabs: { id: View; key: string }[] = [
    { id: 'chat', key: 'nav_chat' }, { id: 'search', key: 'nav_search' }, { id: 'stats', key: 'nav_stats' }, { id: 'knowledge', key: 'nav_knowledge' },
  ]
  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="flex items-center gap-4 px-4 h-14 bg-white border-b border-slate-200">
        <h1 className="font-bold text-lg tracking-tight">GroundRAG</h1>
        <span className="text-xs text-slate-400 hidden sm:inline">{t('subtitle')}</span>
        <div className="ml-auto flex items-center gap-3">
          <LanguageToggle />
          <nav className="flex gap-1">
            {tabs.map((tb) => (
              <button key={tb.id} onClick={() => setView(tb.id)}
                className={`px-3 py-1.5 rounded text-sm ${view === tb.id ? 'bg-blue-600 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>
                {t(tb.key)}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main>
        {view === 'chat' && <ChatView demoUsers={demoUsers} currentUser={currentUser} loginAs={loginAs} />}
        {view === 'search' && <SearchView />}
        {view === 'stats' && <StatsView />}
        {view === 'knowledge' && <KnowledgeBaseView demoUsers={demoUsers} currentUser={currentUser} loginAs={loginAs} />}
      </main>
    </div>
  )
}

export default function App() {
  return <LangProvider><Shell /></LangProvider>
}
