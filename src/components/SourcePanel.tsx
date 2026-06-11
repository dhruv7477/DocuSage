import { useEffect, useMemo, useState } from 'react'
import { Viewer, Worker } from '@react-pdf-viewer/core'
import { searchPlugin } from '@react-pdf-viewer/search'
import { X, Loader2, AlertTriangle, FileText } from 'lucide-react'
import { documents as docsApi } from '@/api'
import { getApiErrorMessage } from '@/api/client'
import { useSourcePanel } from '@/store/source-panel'

import '@react-pdf-viewer/core/lib/styles/index.css'
import '@react-pdf-viewer/search/lib/styles/index.css'

// Use the pdfjs worker that ships with pdfjs-dist@3.11.174 to match
// @react-pdf-viewer/core@3.12 (newer pdfjs versions break the viewer).
const PDF_WORKER_URL = 'https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js'

function isPdf(filename: string): boolean {
  return /\.pdf$/i.test(filename)
}

function buildHighlightKeyword(text: string): string {
  // The PDF search plugin matches on a literal keyword. A long chunk
  // rarely matches verbatim (line breaks, hyphenation), so we trim to
  // a distinctive ~80-char prefix from the first non-trivial sentence.
  const cleaned = text.replace(/\s+/g, ' ').trim()
  // Skip leading boilerplate like headings if any.
  return cleaned.slice(0, 80)
}

/**
 * Thin gate component — subscribes only to `open` so it can short-circuit
 * to `null` cheaply. Only when the panel is actually open do we mount the
 * heavy content component (which kicks off the bytes fetch, PDF worker
 * load, etc.). Splitting the component this way guarantees that React
 * never sees a varying hook count on the SourcePanel instance.
 */
export function SourcePanel() {
  const open = useSourcePanel((s) => s.open)
  if (!open) return null
  return <SourcePanelContent />
}

function SourcePanelContent() {
  const source = useSourcePanel((s) => s.source)
  const close = useSourcePanel((s) => s.close)
  const [bytes, setBytes] = useState<Uint8Array | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Esc to close
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [close])

  // Fetch document bytes whenever the source changes.
  useEffect(() => {
    if (!source) return
    let cancelled = false
    setBytes(null)
    setError(null)
    setLoading(true)
    docsApi.file(source.document_id)
      .then((res) => {
        if (cancelled) return
        setBytes(new Uint8Array(res.data))
      })
      .catch((err) => {
        if (cancelled) return
        setError(getApiErrorMessage(err, 'Could not load the document.'))
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [source])

  // Build a fresh search plugin per source so its internal state resets.
  const searchPluginInstance = useMemo(() => searchPlugin({
    keyword: source ? buildHighlightKeyword(source.text ?? '') : '',
  }), [source])

  // Auto-jump to page & trigger highlight once the document is rendered.
  const initialPage = source?.page != null && source.page > 0 ? source.page - 1 : 0

  const filename = source?.source ?? 'Document'
  const canRenderPdf = source && isPdf(filename) && bytes

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
        className="fixed top-0 right-0 z-50 h-full w-full sm:w-[min(720px,90vw)] bg-panel border-l border-border shadow-2xl flex flex-col animate-slide-in-right"
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
                Page {source.page}
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
        <div className="flex-1 overflow-hidden bg-base/[0.02]">
          {loading && (
            <div className="h-full flex items-center justify-center text-dim">
              <Loader2 className="w-5 h-5 animate-spin mr-2" />
              Loading document…
            </div>
          )}

          {error && (
            <div className="h-full flex items-center justify-center p-6">
              <div className="flex items-start gap-3 max-w-md p-4 bg-red/8 border border-red/30 rounded-lg text-sm text-base">
                <AlertTriangle className="w-4 h-4 text-red shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            </div>
          )}

          {!loading && !error && source && !isPdf(filename) && (
            <div className="h-full overflow-y-auto p-6 text-sm leading-relaxed whitespace-pre-wrap text-base/90">
              <p className="font-mono text-2xs text-dim uppercase tracking-wider mb-3">
                Plain-text preview (non-PDF source)
              </p>
              <div className="p-4 bg-base/5 border-l-2 border-teal rounded">
                {source.text || <span className="italic text-dim">No preview text available.</span>}
              </div>
            </div>
          )}

          {!loading && !error && canRenderPdf && (
            <Worker workerUrl={PDF_WORKER_URL}>
              <Viewer
                fileUrl={bytes}
                plugins={[searchPluginInstance]}
                initialPage={initialPage}
              />
            </Worker>
          )}
        </div>
      </aside>
    </>
  )
}
