import { useState } from 'react'
import { search, type SearchResult } from '../api/client'
import { SearchResults } from '../components/SearchResults'
import { useLang } from '../i18n'

export function SearchView() {
  const { t } = useLang()
  const [q, setQ] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const run = async () => {
    if (!q.trim()) return
    setLoading(true); setErr('')
    try { setResults(await search(q)) }
    catch (e) { setErr((e as Error).message) }
    finally { setLoading(false) }
  }

  return (
    <div className="p-3">
      <div className="flex gap-2 mb-2">
        <input value={q} onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && run()}
          placeholder={t('search_placeholder')}
          className="flex-1 border border-slate-300 rounded px-3 py-2 outline-none focus:border-blue-500" />
        <button onClick={run} disabled={loading}
          className="px-4 py-2 rounded bg-slate-700 text-white disabled:bg-slate-300">
          {loading ? t('searching') : t('search_btn')}
        </button>
      </div>
      {err && <div className="text-red-600 text-sm">⚠ {err}</div>}
      <SearchResults results={results} />
    </div>
  )
}
