import { useState, useRef, useEffect, useCallback } from 'react'
import { ExternalLink } from 'lucide-react'
import type { CitedSource } from '@/api'
import { useSourcePanel } from '@/store/source-panel'

interface CitationLensProps {
  source: CitedSource
  query?: string
  children: React.ReactNode
}

const STOP_WORDS = new Set([
  'the', 'and', 'for', 'that', 'this', 'with', 'from', 'what', 'when', 'where',
  'which', 'have', 'has', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
  'about', 'into', 'than', 'then', 'them', 'they', 'their', 'there',
])

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function highlightQueryTerms(text: string, query: string): React.ReactNode[] {
  if (!query.trim()) return [text]

  const terms = Array.from(
    new Set(
      query
        .toLowerCase()
        .split(/\s+/)
        .filter(t => t.length > 2 && !STOP_WORDS.has(t))
        .map(escapeRegex)
    )
  )

  if (!terms.length) return [text]

  const re = new RegExp(`(${terms.join('|')})`, 'gi')
  const parts: React.ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  let i = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index))
    parts.push(
      <mark
        key={`mk-${i++}-${m.index}`}
        className="bg-amber/30 text-base rounded px-0.5"
      >
        {m[0]}
      </mark>
    )
    last = m.index + m[0].length
  }
  if (last < text.length) parts.push(text.slice(last))
  return parts
}

export function CitationLens({ source, query, children }: CitationLensProps) {
  const openSource = useSourcePanel((s) => s.openSource)
  const [open, setOpen] = useState(false)
  const wrapperRef = useRef<HTMLSpanElement>(null)
  const hoverTimerRef = useRef<number | null>(null)

  const cancelHoverClose = useCallback(() => {
    if (hoverTimerRef.current != null) {
      window.clearTimeout(hoverTimerRef.current)
      hoverTimerRef.current = null
    }
  }, [])

  const scheduleClose = useCallback(() => {
    cancelHoverClose()
    hoverTimerRef.current = window.setTimeout(() => setOpen(false), 120)
  }, [cancelHoverClose])

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
      cancelHoverClose()
    }
  }, [open, cancelHoverClose])

  const heading = source.heading_path && source.heading_path.length
    ? source.heading_path.join(' › ')
    : null

  return (
    <span
      ref={wrapperRef}
      className="relative inline-block"
      onMouseEnter={cancelHoverClose}
      onMouseLeave={scheduleClose}
    >
      <span
        onMouseEnter={() => { cancelHoverClose(); setOpen(true) }}
        onClick={(e) => { e.stopPropagation(); setOpen(o => !o) }}
      >
        {children}
      </span>

      {open && (
        <div
          role="dialog"
          aria-label={`Source ${source.source}, page ${source.page ?? '?'}`}
          className="absolute z-50 left-0 top-full mt-1 w-[min(420px,calc(100vw-2rem))] bg-panel border border-border rounded-lg shadow-2xl p-3 animate-fade-up text-left cursor-default"
        >
          {/* Header */}
          <div className="flex items-baseline gap-2 mb-2 pb-2 border-b border-border/60">
            <span className="font-mono text-teal/90 text-2xs font-bold truncate" title={source.source}>
              {source.source}
            </span>
            {source.page != null && (
              <span className="font-mono text-2xs text-dim shrink-0">p.{source.page}</span>
            )}
            <span className="font-mono text-2xs text-dim ml-auto shrink-0">
              {(source.score * 100).toFixed(0)}%
            </span>
          </div>

          {/* Heading path */}
          {heading && (
            <p
              className="font-mono text-2xs text-dim mb-2 truncate"
              title={heading}
            >
              {heading}
            </p>
          )}

          {/* Chunk text with query highlights */}
          <div className="text-xs leading-relaxed text-base/90 max-h-48 overflow-y-auto whitespace-pre-wrap break-words pr-1">
            {source.text
              ? highlightQueryTerms(source.text, query ?? '')
              : <span className="text-dim italic">No preview available for this chunk.</span>
            }
          </div>

          {/* Footer */}
          <button
            onClick={() => { setOpen(false); openSource(source) }}
            className="mt-2 pt-2 border-t border-border/60 w-full flex items-center justify-center gap-1.5 text-xs text-teal/90 hover:text-teal transition-colors"
          >
            Open in viewer
            <ExternalLink className="w-3 h-3" />
          </button>
        </div>
      )}
    </span>
  )
}
