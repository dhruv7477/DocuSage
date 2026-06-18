import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import type { ThinkingStage } from '@/hooks/useSSE'

/**
 * Claude Code-style status line: spinner + rotating playful phrase per
 * pipeline stage + elapsed-seconds counter. The phrase rotates every
 * 2.5 s, picked deterministically from elapsed time so re-renders never
 * cause flicker.
 */

const PHRASES: Record<Exclude<ThinkingStage, null>, string[]> = {
  analyzing: [
    'Reading your question…',
    'Sizing up the request…',
    'Figuring out what you need…',
  ],
  routing: [
    'Choosing the best approach…',
    'Picking the right tool…',
  ],
  retrieving: [
    'Combing through your documents…',
    'Spelunking the archives…',
    'Hunting down relevant passages…',
    'Leafing through the pages…',
  ],
  reranking: [
    'Weighing the evidence…',
    'Sorting wheat from chaff…',
    'Ranking the best matches…',
  ],
  generating: [
    'Crafting your answer…',
    'Connecting the dots…',
    'Putting it into words…',
  ],
  summarizing: [
    'Distilling the essence…',
    'Boiling it down…',
    'Reading the whole thing so you don’t have to…',
  ],
  validating: [
    'Double-checking citations…',
    'Verifying every claim…',
  ],
}

const ROTATE_SECONDS = 2.5

interface ThinkingStatusProps {
  stage: ThinkingStage
  startedAt: number | null
}

export function ThinkingStatus({ stage, startedAt }: ThinkingStatusProps) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])

  if (!stage) {
    return (
      <span className="flex items-center gap-2 text-dim text-sm">
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        Thinking…
      </span>
    )
  }

  const elapsed = startedAt ? Math.max(0, Math.floor((now - startedAt) / 1000)) : 0
  const phrases = PHRASES[stage]
  const phrase = phrases[Math.floor(elapsed / ROTATE_SECONDS) % phrases.length]

  return (
    <span className="flex items-center gap-2 text-dim text-sm">
      <Loader2 className="w-3.5 h-3.5 animate-spin text-teal" />
      <span className="text-subtle">{phrase}</span>
      <span className="font-mono text-2xs text-dim">· {elapsed}s</span>
    </span>
  )
}
