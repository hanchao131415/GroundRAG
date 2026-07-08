import { useState } from 'react'
import { ChatView } from './views/ChatView'
import { SearchView } from './views/SearchView'
import { StatsView } from './views/StatsView'

type View = 'chat' | 'search' | 'stats'

export default function App() {
  const [view, setView] = useState<View>('chat')
  const tabs: { id: View; label: string }[] = [
    { id: 'chat', label: '💬 问答' },
    { id: 'search', label: '🔍 检索' },
    { id: 'stats', label: '📊 统计' },
  ]
  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="flex items-center gap-4 px-4 h-14 bg-white border-b border-slate-200">
        <h1 className="font-bold text-lg tracking-tight">GroundRAG</h1>
        <span className="text-xs text-slate-400 hidden sm:inline">企业级 RAG 知识库 · 混合检索 + RBAC + 全链路可观测</span>
        <nav className="ml-auto flex gap-1">
          {tabs.map((t) => (
            <button key={t.id} onClick={() => setView(t.id)}
              className={`px-3 py-1.5 rounded text-sm ${view === t.id ? 'bg-blue-600 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {view === 'chat' && <ChatView />}
        {view === 'search' && <SearchView />}
        {view === 'stats' && <StatsView />}
      </main>
    </div>
  )
}
