import { useState } from 'react'
import type { Trace } from '../types'
import { useLang } from '../i18n'

function fmtMs(ms: number): string {
  if (ms <= 0 && ms >= 0) return '<0.1ms'
  if (ms < 0.1) return `${(ms * 1000).toFixed(0)}μs`
  if (ms < 1) return `${ms.toFixed(2)}ms`
  return `${ms.toFixed(1)}ms`
}

export function TracePanel({ trace }: { trace: Trace }) {
  const { t } = useLang()
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-2 text-xs">
      <button onClick={() => setOpen((o) => !o)} className="text-slate-500 hover:text-slate-800">
        {open ? '▾' : '▸'} {t('trace_label')} · {fmtMs(trace.total_ms)} · {t('tokens_label')} {trace.tokens.total} · ≈${trace.cost_usd.toFixed(6)}
      </button>
      {open && (
        <div className="mt-1 font-mono text-[11px] text-slate-600 border-l-2 border-slate-200 pl-2">
          {trace.steps.map((s, i) => (
            <div key={i}>└ {s.name.padEnd(6)} {fmtMs(s.ms)}{s.tokens ? ` ${t('tok_label')}:${s.tokens.total}` : ''}</div>
          ))}
          <div className="text-slate-400 mt-1">{t('trace_id_label')}: {trace.trace_id}</div>
        </div>
      )}
    </div>
  )
}
