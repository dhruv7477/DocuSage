import { useEffect, useState } from 'react'
import { X, Loader2, AlertTriangle, FileText, ChevronLeft, ChevronRight, Info } from 'lucide-react'
import { useSourcePanel } from '@/store/source-panel'
import { usePageImage } from '@/hooks/usePageImage'
import { highlightChunkId } from './citations'

function isPdf(filename: string): boolean {
  return /\.pdf$/i.test(filename)
}

/**
 * Thin gate component — subscribes only to `open` so it can short-circuit
 * to `null` cheaply. The heavy content component (image fetch, page
 * state) only mounts while the panel is open, so React never sees a
 * varying hook count on one component instance.
 */
export function SourcePanel() {
  const open = useSourcePanel((s) => s.open)
  if (!open) return null
  return <SourcePanelContent />
}

function SourcePanelContent() {
  const source = useSourcePanel((s) => s.source)
  const close = useSourcePanel((s) => s.close)

  const citedPage = source?.page && source.page >= 1 ? source.page : 1
  const [page, setPage] = useState(citedPage)

  // Jump back to the cited page whenever a new source opens.
  useEffect(() => setPage(citedPage), [citedPage, source?.chunk_id])

  // Esc to close
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [close])

  const filename = source?.source ?? 'Document'
  const pdf = source && isPdf(filename)

  // Only highlight on the cited page — other pages render plain.
  const { data, isLoading, isError } = usePageImage(
    pdf ? source.document_id : undefined,
    page,
    page === citedPage && source ? highlightChunkId(source) : undefined,
    2,
  )

  const pageCount = data?.pageCount ?? 0

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={close}
        className="fixed inset-0 z-40 bg-black/40 animate-fade-up"
      />

      {/* Drawer */}
      <aside
        role="dialog"
        aria-label={`Source viewer: ${filename}`}
        className="fixed top-0 right-0 z-50 h-full w-full sm:w-[min(720px,90vw)] bg-panel/80 backdrop-blur-2xl border-l border-border shadow-2xl flex flex-col animate-slide-in-right"
      >
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border shrink-0">
          <FileText className="w-4 h-4 text-teal shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-base truncate" title={filename}>
              {filename}
            </p>
            {source?.page != null && (
              <p className="font-mono text-2xs text-dim">
                Cited on page {source.page}
                {source.heading_path && source.heading_path.length > 0 && (
                  <span> · {source.heading_path.join(' › ')}</span>
                )}
              </p>
            )}
          </div>
          <button
            onClick={close}
            aria-label="Close source viewer"
            className="p-1.5 rounded hover:bg-base/5 text-dim hover:text-base transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto bg-base/[0.02]">
          {!pdf && source && (
            <div className="h-full overflow-y-auto p-6 text-sm leading-relaxed whitespace-pre-wrap text-base/90">
              <p className="font-mono text-2xs text-dim uppercase tracking-wider mb-3">
                Plain-text preview (non-PDF source)
              </p>
              <div className="p-4 bg-base/5 border-l-2 border-teal rounded">
                {source.text || <span className="italic text-dim">No preview text available.</span>}
              </div>
            </div>
          )}

          {pdf && isLoading && (
            <div className="h-full flex items-center justify-center text-dim">
              <Loader2 className="w-5 h-5 animate-spin mr-2" />
              Rendering page…
            </div>
          )}

          {pdf && isError && (
            <div className="h-full flex items-center justify-center p-6">
              <div className="flex items-start gap-3 max-w-md p-4 bg-red/8 border border-red/30 rounded-lg text-sm text-base">
                <AlertTriangle className="w-4 h-4 text-red shrink-0 mt-0.5" />
                <span>Could not render this page. Try downloading the document instead.</span>
              </div>
            </div>
          )}

          {pdf && data && (
            <div className="p-4">
              {page === citedPage && !data.highlightFound && (
                <div className="flex items-center gap-2 mb-3 px-3 py-2 bg-amber/8 border border-amber/25 rounded-lg text-xs text-subtle">
                  <Info className="w-3.5 h-3.5 text-amber shrink-0" />
                  Exact passage couldn't be marked on this page — showing the full cited page.
                </div>
              )}
              <img
                src={data.url}
                alt={`${filename} — page ${page}`}
                className="w-full rounded-lg border border-border shadow-panel"
              />
            </div>
          )}
        </div>

        {/* Footer: page navigation */}
        {pdf && pageCount > 0 && (
          <div className="flex items-center justify-center gap-3 px-4 py-3 border-t border-border shrink-0">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              aria-label="Previous page"
              className="p-1.5 rounded hover:bg-base/5 text-subtle hover:text-base disabled:opacity-30 disabled:pointer-events-none transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="font-mono text-xs text-subtle">
              Page {page} / {pageCount}
              {page !== citedPage && (
                <button
                  onClick={() => setPage(citedPage)}
                  className="ml-2 text-teal/80 hover:text-teal transition-colors"
                >
                  back to citation
                </button>
              )}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
              disabled={page >= pageCount}
              aria-label="Next page"
              className="p-1.5 rounded hover:bg-base/5 text-subtle hover:text-base disabled:opacity-30 disabled:pointer-events-none transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </aside>
    </>
  )
}
