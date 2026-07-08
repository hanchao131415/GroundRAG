import { useLang } from '../i18n'
import type { Source } from '../types'

export function SourceCard({ s, index }: { s: Source; index: number }) {
  const { t } = useLang()
  return (
    <div className="border border-slate-200 rounded-md p-2 text-sm bg-slate-50">
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-xs text-slate-500">{t('source_doc', { n: index })}</span>
        <span className="font-medium text-slate-700 truncate">{s.source}</span>
        {s.page ? <span className="text-xs text-slate-400">p.{s.page}</span> : null}
        <span className="ml-auto text-xs px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700">
          {s.department}
        </span>
        <span className="text-xs text-slate-400">score {s.score.toFixed(3)}</span>
      </div>
      <p className="text-slate-600 line-clamp-2">{s.preview}</p>
    </div>
  )
}
