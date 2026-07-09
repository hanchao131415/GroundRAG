import { useLang } from '../i18n'
import type { SearchResult } from '../api/client'

export function SearchResults({ results }: { results: SearchResult[] }) {
  const { t } = useLang()
  if (results.length === 0) return <div className="text-slate-400 text-sm p-4">{t('no_results')}</div>
  return (
    <div className="space-y-2 p-3">
      {results.map((r, i) => (
        <div key={i} className="border border-slate-200 rounded p-2 text-sm">
          <div className="flex gap-2 mb-1">
            <span className="font-medium text-slate-700 truncate">{r.source}</span>
            <span className="ml-auto text-xs text-slate-400">{t('score')} {r.score.toFixed(3)}</span>
          </div>
          <p className="text-slate-600">{r.preview}</p>
        </div>
      ))}
    </div>
  )
}
