import { useEffect, useState } from 'react'
import { getStats } from '../api/client'
import { useLang } from '../i18n'

export function StatsView() {
  const { t } = useLang()
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    getStats().then(setStats).catch((e) => setErr((e as Error).message))
  }, [])

  if (err) return <div className="p-4 text-red-600">⚠ {err}</div>
  if (!stats) return <div className="p-4 text-slate-400">{t('stats_loading')}</div>

  const kb = (stats.knowledge_base ?? {}) as Record<string, unknown>
  const cache = (stats.cache ?? {}) as Record<string, unknown>

  return (
    <div className="p-4 space-y-3 text-sm">
      <div className="grid grid-cols-2 gap-2">
        <Card label={t('stat_docs')} value={String(kb.total_documents ?? '-')} />
        <Card label={t('stat_chunks')} value={String(kb.total_chunks ?? '-')} />
        <Card label={t('stat_avg')} value={String(Math.round(Number(kb.avg_chunk_size ?? 0)))} />
        <Card label={t('stat_cache')} value={String(cache.total ?? 0)} />
      </div>
      <div>
        <div className="text-slate-500 mb-1">{t('dept_dist')}</div>
        <pre className="bg-slate-50 border border-slate-200 rounded p-2 text-xs">
{JSON.stringify(kb.departments ?? {}, null, 2)}
        </pre>
      </div>
    </div>
  )
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-slate-200 rounded p-3">
      <div className="text-2xl font-semibold text-slate-800">{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}
