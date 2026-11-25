import { useState, useRef, useEffect, useLayoutEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { ExternalLink } from 'lucide-react'
import type { CitedSource } from '@/api'
import { useSourcePanel } from '@/store/source-panel'
import { usePageImage } from '@/hooks/usePageImage'
import { highlightChunkId } from './citations'

function isPdf(filename: string): boolean {
  return /\.pdf$/i.test(filename)
}

/** Small server-rendered page screenshot with the chunk highlighted. */
function LensThumbnail({ source }: { source: CitedSource }) {
  const { data, isLoading, isError } = usePageImage(
    source.document_id,
    source.page && source.page >= 1 ? source.page : 1,
    highlightChunkId(source),
    1.5,  // thumbnail scale — small + fast
    true, // crop to just the cited passage
  )

  if (isError) return null
  if (isLoading) {
    return (
      <div className="w-full h-32 mb-2 rounded bg-base/5 border border-border/60 animate-pulse" />
    )
  }
  if (!data) return null
  return (
    <img
      src={data.url}
      alt={`${source.source} — page ${source.page ?? 1}`}
      className="w-full h-auto mb-2 rounded border border-teal/20"
    />
  )
}

interface CitationLensProps {
  source: CitedSource
  query?: string
  children: React.ReactNode
  /** Render the trigger as a full-width block (for source-list rows)
   *  instead of an inline chip. */
  block?: boolean
  /** The source's rank among the answer's passages (1 = most relevant),
   *  shown instead of the reranker score, which is not calibrated. */
  rank?: number
  total?: number
}

const STOP_WORDS = new Set([
  'the', 'and', 'for', 'that', 'this', 'with', 'from', 'what', 'when', 'where',
  'which', 'have', 'has', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
  'about', 'into', 'than', 'then', 'them', 'they', 'their', 'there',
])

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export function highlightQueryTerms(text: string, query: string): React.ReactNode[] {
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

// Anchored, viewport-clamped position for the portaled popover.
interface PopoverPos {
  left: number
  width: number
  top?: number
  bottom?: number
  maxHeight: number
}

export function CitationLens({ source, query, children, block = false, rank, total }: CitationLensProps) {
  const openSource = useSourcePanel((s) => s.openSource)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<PopoverPos | null>(null)
  const wrapperRef = useRef<HTMLSpanElement>(null)
  const popoverRef = useRef<HTMLDivElement>(null)
  const hoverTimerRef = useRef<number | null>(null)

  const cancelHoverClose = useCallback(() => {
    if (hoverTimerRef.current != null) {
      window.clearTimeout(hoverTimerRef.current)
      hoverTimerRef.current = null
    }
  }, [])

  const scheduleClose = useCallback(() => {
    cancelHoverClose()
    hoverTimerRef.current = window.setTimeout(() => setOpen(false), 140)
  }, [cancelHoverClose])

  // Compute a fixed position from the chip's rect. Flips above the chip
  // when there isn't enough room below, and clamps within the viewport.
  const computePos = useCallback(() => {
    const el = wrapperRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const margin = 12
    const width = Math.min(420, window.innerWidth - margin * 2)
    let left = r.left
    if (left + width > window.innerWidth - margin) left = window.innerWidth - margin - width
    if (left < margin) left = margin

    const spaceBelow = window.innerHeight - r.bottom - margin
    const spaceAbove = r.top - margin
    if (spaceBelow >= 240 || spaceBelow >= spaceAbove) {
      setPos({ left, width, top: r.bottom + 6, maxHeight: spaceBelow })
    } else {
      setPos({ left, width, bottom: window.innerHeight - r.top + 6, maxHeight: spaceAbove })
    }
  }, [])

  useLayoutEffect(() => {
    if (!open) return
    computePos()
    // capture-phase scroll catches scrolling inside any container, not just window
    window.addEventListener('scroll', computePos, true)
    window.addEventListener('resize', computePos)
    return () => {
      window.removeEventListener('scroll', computePos, true)
      window.removeEventListener('resize', computePos)
    }
  }, [open, computePos])

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      const t = e.target as Node
      if (
        wrapperRef.current && !wrapperRef.current.contains(t) &&
        popoverRef.current && !popoverRef.current.contains(t)
      ) {
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
      className={block ? 'relative block' : 'relative inline-block'}
      onMouseEnter={cancelHoverClose}
      onMouseLeave={scheduleClose}
    >
      <span
        className={block ? 'block' : undefined}
        onMouseEnter={() => { cancelHoverClose(); setOpen(true) }}
        onClick={(e) => { e.stopPropagation(); setOpen(o => !o) }}
      >
        {children}
      </span>

      {open && pos && createPortal(
        <div
          ref={popoverRef}
          role="dialog"
          aria-label={`Source ${source.source}, page ${source.page ?? '?'}`}
          onMouseEnter={cancelHoverClose}
          onMouseLeave={scheduleClose}
          style={{
            position: 'fixed',
            left: pos.left,
            width: pos.width,
            top: pos.top,
            bottom: pos.bottom,
            maxHeight: pos.maxHeight,
          }}
          className="z-[100] overflow-y-auto bg-panel border border-teal/25 rounded-lg shadow-hud p-3 animate-fade-up text-left cursor-default"
        >
          {/* Header */}
          <div className="flex items-baseline gap-2 mb-2 pb-2 border-b border-border/60">
            <span className="font-mono text-teal/90 text-2xs font-bold truncate" title={source.source}>
              {source.source}
            </span>
            {source.page != null && (
              <span className="font-mono text-2xs text-dim shrink-0">p.{source.page}</span>
            )}
            {rank != null && total != null && (
              <span className="font-mono text-2xs text-dim ml-auto shrink-0">
                rank {rank} of {total}
              </span>
            )}
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

          {/* Page screenshot with the passage marked (PDF only) */}
          {isPdf(source.source) && <LensThumbnail source={source} />}

          {/* Chunk text with query highlights */}
          <div className={
            isPdf(source.source)
              ? "text-xs leading-relaxed text-base/90 max-h-20 overflow-y-auto whitespace-pre-wrap break-words pr-1"
              : "text-xs leading-relaxed text-base/90 max-h-48 overflow-y-auto whitespace-pre-wrap break-words pr-1"
          }>
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
        </div>,
        document.body,
      )}
    </span>
  )
}
