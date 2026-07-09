import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'

export type Lang = 'zh' | 'en'

const dict: Record<Lang, Record<string, string>> = {
  zh: {
    // App shell
    subtitle: '企业级 RAG 知识库 · 混合检索 + RBAC + 全链路可观测',
    nav_chat: '💬 问答',
    nav_search: '🔍 检索',
    nav_stats: '📊 统计',

    // UserSwitcher
    current_user: '当前用户：',
    not_logged_in: '未登录（全库）',
    visible_depts: '可见部门：',
    no_filter: '未登录 · 全库可见',

    // ChatWindow
    input_placeholder: '输入问题…',
    send: '发送',
    generating: '生成中…',
    chat_empty: '试试问「工作满3年年假几天」「住宿费报销上限」「密码多久换一次」',

    // MessageBubble
    assistant_loading: '…',
    error_prefix: '⚠',
    sources_label: '引用来源',

    // SourceCard
    source_doc: '[文档{n}]',
    page: 'p.{n}',
    score: '相关度',

    // TracePanel
    trace_label: 'trace',
    trace_id_label: 'trace_id',
    tokens_label: 'tokens',
    tok_label: 'tok',

    // SearchView / SearchResults
    search_placeholder: '纯检索（不调 LLM）…',
    search_btn: '检索',
    searching: '检索中…',
    no_results: '无召回',

    // StatsView
    stats_loading: '加载中…',
    stat_docs: '文档数',
    stat_chunks: '分块数',
    stat_avg: '平均块大小',
    stat_cache: '缓存条目',
    dept_dist: '部门分布',

    // LanguageToggle
    lang_zh: '中',
    lang_en: 'EN',
  },
  en: {
    // App shell
    subtitle: 'Enterprise RAG · Hybrid Retrieval + RBAC + Full-pipeline Observability',
    nav_chat: '💬 Chat',
    nav_search: '🔍 Search',
    nav_stats: '📊 Stats',

    // UserSwitcher
    current_user: 'User: ',
    not_logged_in: 'Not logged in (all docs)',
    visible_depts: 'Visible depts: ',
    no_filter: 'Not logged in · all docs',

    // ChatWindow
    input_placeholder: 'Ask a question…',
    send: 'Send',
    generating: 'Generating…',
    chat_empty: 'Try "工作满3年年假几天" / "住宿费报销上限" / "密码多久换一次"',

    // MessageBubble
    assistant_loading: '…',
    error_prefix: '⚠',
    sources_label: 'Sources',

    // SourceCard
    source_doc: '[Doc {n}]',
    page: 'p.{n}',
    score: 'Score',

    // TracePanel
    trace_label: 'trace',
    trace_id_label: 'trace_id',
    tokens_label: 'tokens',
    tok_label: 'tok',

    // SearchView / SearchResults
    search_placeholder: 'Search only (no LLM)…',
    search_btn: 'Search',
    searching: 'Searching…',
    no_results: 'No results',

    // StatsView
    stats_loading: 'Loading…',
    stat_docs: 'Documents',
    stat_chunks: 'Chunks',
    stat_avg: 'Avg chunk size',
    stat_cache: 'Cache entries',
    dept_dist: 'Department distribution',

    // LanguageToggle
    lang_zh: '中',
    lang_en: 'EN',
  },
}

interface Ctx { lang: Lang; setLang: (l: Lang) => void; t: (key: string, p?: Record<string, string | number>) => string }
const LangContext = createContext<Ctx | null>(null)

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => (localStorage.getItem('groundrag_lang') as Lang) || 'zh')
  const setLang = useCallback((l: Lang) => { setLangState(l); localStorage.setItem('groundrag_lang', l) }, [])
  const t = useCallback((key: string, p?: Record<string, string | number>) => {
    let s = dict[lang][key] ?? key
    if (p) for (const k of Object.keys(p)) s = s.replace(`{${k}}`, String(p[k]))
    return s
  }, [lang])
  return <LangContext.Provider value={{ lang, setLang, t }}>{children}</LangContext.Provider>
}

export function useLang() {
  const ctx = useContext(LangContext)
  if (!ctx) throw new Error('useLang must be used within LangProvider')
  return ctx
}

export function LanguageToggle() {
  const { lang, setLang, t } = useLang()
  return (
    <div className="flex gap-1 text-xs">
      <button onClick={() => setLang('zh')} className={lang === 'zh' ? 'font-bold text-slate-900' : 'text-slate-400'}>{t('lang_zh')}</button>
      <span className="text-slate-300">/</span>
      <button onClick={() => setLang('en')} className={lang === 'en' ? 'font-bold text-slate-900' : 'text-slate-400'}>{t('lang_en')}</button>
    </div>
  )
}
