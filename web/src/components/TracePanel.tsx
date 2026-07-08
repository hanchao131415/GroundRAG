import { useState } from 'react'
import type { Trace } from '../types'

function fmtMs(ms: number): string {
  if (ms <= 0 && ms >= 0) return '<0.1ms'
  if (ms < 0.1) return `${(ms * 1000).toFixed(0)}μs`
  if (ms < 1) return `${ms.toFixed(2)}ms`
  return `${ms.toFixed(1)}ms`
}

export function TracePanel({ trace }: { trace: Trace }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-2 text-xs">
      <button onClick={() => setOpen((o) => !o)} className="text-slate-500 hover:text-slate-800">
        {open ? '▾' : '▸'} trace · {fmtMs(trace.total_ms)} · tokens {trace.tokens.total} · ≈${trace.cost_usd.toFixed(6)}
      </button>
      {open && (
        <div className="mt-1 font-mono text-[11px] text-slate-600 border-l-2 border-slate-200 pl-2">
          {trace.steps.map((s, i) => (
            <div key={i}>└ {s.name.padEnd(6)} {fmtMs(s.ms)}{s.tokens ? ` tok:${s.tokens.total}` : ''}</div>
          ))}
          <div className="text-slate-400 mt-1">trace_id: {trace.trace_id}</div>
        </div>
      )}
    </div>
  )
}
