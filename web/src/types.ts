export interface Source {
  source: string
  page: number | null
  department: string
  score: number
  preview: string
}

export interface TokenUsage { prompt: number; completion: number; total: number }

export interface TraceStep {
  name: string
  ms: number
  tokens?: TokenUsage
}

export interface Trace {
  trace_id: string
  steps: TraceStep[]
  total_ms: number
  tokens: TokenUsage
  cost_usd: number
}

export interface DemoUser {
  user_id: string
  name: string
  departments: string[]
  role: string
}

export type SSEEvent =
  | { type: 'sources'; items: Source[] }
  | { type: 'token'; text: string }
  | { type: 'trace'; trace: Trace }
  | { type: 'error'; message: string }
  | { type: 'done' }
