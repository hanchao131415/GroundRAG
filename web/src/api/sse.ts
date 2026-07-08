import type { SSEEvent } from '../types'

/**
 * 解析 SSE 文本缓冲：按 "\n\n" 分帧，完整帧解析为事件，最后一帧（可能不完整）作为 rest 返回。
 * 纯函数，无副作用，便于测试。
 */
export function parseSSEChunk(buffer: string): { events: SSEEvent[]; rest: string } {
  const events: SSEEvent[] = []
  const frames = buffer.split('\n\n')
  const rest = frames.pop() ?? ''

  for (const frame of frames) {
    const line = frame.split('\n').find((l) => l.startsWith('data:'))
    if (!line) continue
    const payload = line.slice(5).trim()
    if (payload === '[DONE]') {
      events.push({ type: 'done' })
      continue
    }
    try {
      const obj = JSON.parse(payload)
      if (obj && typeof obj.type === 'string') events.push(obj as SSEEvent)
    } catch {
      // 忽略损坏帧
    }
  }
  return { events, rest }
}
