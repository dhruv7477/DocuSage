import { useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useDropzone } from 'react-dropzone'
import {
  Upload, FileText, AlertCircle, CheckCircle2,
  Clock, Loader2, Trash2, X, Database
} from 'lucide-react'
import { clsx } from 'clsx'
import { formatDistanceToNow } from 'date-fns'
import toast from 'react-hot-toast'
import { documents as docsApi, IN_FLIGHT_STATUSES, type Document, type DocumentStatus } from '@/api'
import { getApiErrorMessage } from '@/api/client'
import { useApp } from '@/store/app'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'

// ── Status config ─────────────────────────────────────────────────
const STATUS: Record<DocumentStatus, {
  label: string
  icon: typeof Clock
  color: string
  bg: string
  spin?: boolean
}> = {
  pending:    { label: 'Queued',     icon: Clock,        color: 'text-amber',  bg: 'bg-amber/10 border-amber/20' },
  processing: { label: 'Indexing',   icon: Loader2,      color: 'text-teal',   bg: 'bg-teal/10 border-teal/20', spin: true },
  parsing:    { label: 'Parsing',    icon: Loader2,      color: 'text-teal',   bg: 'bg-teal/10 border-teal/20', spin: true },
  chunking:   { label: 'Chunking',   icon: Loader2,      color: 'text-teal',   bg: 'bg-teal/10 border-teal/20', spin: true },
  embedding:  { label: 'Embedding',  icon: Loader2,      color: 'text-teal',   bg: 'bg-teal/10 border-teal/20', spin: true },
  indexing:   { label: 'Indexing',   icon: Loader2,      color: 'text-teal',   bg: 'bg-teal/10 border-teal/20', spin: true },
  completed:  { label: 'Ready',      icon: CheckCircle2, color: 'text-green',  bg: 'bg-green/10 border-green/20' },
  failed:     { label: 'Failed',     icon: AlertCircle,  color: 'text-red',    bg: 'bg-red/10 border-red/20' },
}

// ── File type icon ────────────────────────────────────────────────
function FileIcon({ type: _type }: { type: string }) {
  return <FileText className="w-4 h-4 text-subtle" />
}

// ── Format file size ──────────────────────────────────────────────
function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / (1024 * 1024)).toFixed(1)} MB`
}

// ── Document row ──────────────────────────────────────────────────
function DocRow({ doc, onDelete }: { doc: Document; onDelete: () => void }) {
  const s = STATUS[doc.status] ?? STATUS.pending
  const Icon = s.icon
  const inFlight = IN_FLIGHT_STATUSES.has(doc.status)
  const progress = Math.max(0, Math.min(100, doc.progress ?? 0))

  return (
    <div className="group flex flex-col px-4 py-3 border-b border-border hover:bg-surface/50 transition-colors">
      <div className="flex items-center gap-3">
        {/* Icon */}
        <div className="w-8 h-8 rounded-lg bg-panel border border-border flex items-center justify-center shrink-0">
          <FileIcon type={doc.file_type} />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <p className="text-sm text-bright font-medium truncate">{doc.original_filename}</p>
          {/* whitespace-nowrap per chip + flex-wrap on the row: in the
              narrow panel the chips flow onto a second line as whole
              units instead of each phrase wrapping mid-text into
              unreadable vertical slivers. */}
          <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 mt-0.5">
            <span className="font-mono text-2xs text-dim whitespace-nowrap">{fmtBytes(doc.file_size_bytes)}</span>
            {doc.chunk_count != null && (
              <span className="font-mono text-2xs text-dim whitespace-nowrap">· {doc.chunk_count} chunks</span>
            )}
            <span className="font-mono text-2xs text-dim whitespace-nowrap">
              · {formatDistanceToNow(new Date(doc.created_at), { addSuffix: true })}
            </span>
          </div>
        </div>

        {/* Status badge */}
        <div className={clsx('flex items-center gap-1.5 px-2 py-1 rounded-full border text-xs shrink-0', s.bg, s.color)}
          title={doc.status === 'failed' && doc.error_message ? doc.error_message : undefined}
        >
          <Icon className={clsx('w-3 h-3', s.spin && 'animate-spin')} />
          {s.label}
          {inFlight && progress > 0 && progress < 100 && (
            <span className="font-mono text-2xs opacity-75 ml-0.5">{progress}%</span>
          )}
        </div>

        {/* Delete */}
        <button
          onClick={onDelete}
          className="opacity-0 group-hover:opacity-100 text-dim hover:text-red transition-all ml-2"
          title="Delete document"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Inline progress bar while ingesting */}
      {inFlight && (
        <div className="mt-2 ml-11 h-0.5 bg-border rounded-full overflow-hidden">
          <div
            className="h-full bg-teal transition-all duration-300 rounded-full"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Error detail */}
      {doc.status === 'failed' && doc.error_message && (
        <p className="text-xs text-red mt-1.5 ml-11 leading-relaxed">{doc.error_message}</p>
      )}
    </div>
  )
}

// ── Upload progress item ──────────────────────────────────────────
interface UploadItem {
  name: string
  progress: number
  done: boolean
  error: boolean
  errorMessage?: string
}

function UploadProgress({ items }: { items: UploadItem[] }) {
  if (items.length === 0) return null
  return (
    <div className="border-b border-border">
      {items.map((item) => (
        <div
          key={item.name}
          className={clsx(
            'flex flex-col gap-1 px-4 py-2.5',
            item.error ? 'bg-red/[0.04]' : 'bg-teal/[0.03]'
          )}
        >
          <div className="flex items-center gap-3">
            <div className={clsx(
              'w-8 h-8 rounded-lg border flex items-center justify-center shrink-0',
              item.error ? 'bg-red/10 border-red/20' : 'bg-teal/10 border-teal/20'
            )}>
              <Upload className={clsx('w-3.5 h-3.5', item.error ? 'text-red' : 'text-teal')} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs text-bright truncate">{item.name}</p>
              {!item.done && !item.error && (
                <div className="mt-1 h-0.5 bg-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-teal transition-all duration-200 rounded-full"
                    style={{ width: `${item.progress}%` }}
                  />
                </div>
              )}
            </div>
            {item.done && <CheckCircle2 className="w-4 h-4 text-green shrink-0" />}
            {item.error && <AlertCircle className="w-4 h-4 text-red shrink-0" />}
            {!item.done && !item.error && (
              <span className="font-mono text-2xs text-dim shrink-0">{item.progress}%</span>
            )}
          </div>
          {item.error && item.errorMessage && (
            <p className="text-2xs text-red ml-11 leading-relaxed">{item.errorMessage}</p>
          )}
        </div>
      ))}
    </div>
  )
}

// ── Main Documents panel ──────────────────────────────────────────
export function DocumentsPanel({ onClose }: { onClose: () => void }) {
  const { activeWorkspaceId } = useApp()
  const qc = useQueryClient()
  const [uploads, setUploads] = useState<UploadItem[]>([])
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ['documents', activeWorkspaceId],
    queryFn: () => docsApi.list(activeWorkspaceId!).then((r) => r.data),
    enabled: !!activeWorkspaceId,
    refetchInterval: (query) => {
      const docs: Document[] = query.state.data?.documents ?? []
      const hasInFlight = docs.some((d) => IN_FLIGHT_STATUSES.has(d.status))
      return hasInFlight ? 4000 : false
    },
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => docsApi.delete(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents', activeWorkspaceId] })
      toast.success('Document deleted')
    },
    onError: (err) => toast.error(getApiErrorMessage(err, 'Delete failed')),
  })

  const onDrop = useCallback(async (files: File[]) => {
    console.log('[DocuSage] onDrop triggered', { fileCount: files.length, activeWorkspaceId })
    if (!activeWorkspaceId) { toast.error('Select a workspace first'); return }

    for (const file of files) {
      console.log('[DocuSage] uploading file', { name: file.name, size: file.size, type: file.type })
      const item: UploadItem = { name: file.name, progress: 0, done: false, error: false }
      setUploads((u) => [...u, item])

      try {
        console.log('[DocuSage] calling docsApi.upload', { workspace: activeWorkspaceId, file: file.name })
        const result = await docsApi.upload(activeWorkspaceId, file, (pct) => {
          console.log('[DocuSage] upload progress', { file: file.name, pct })
          setUploads((u) => u.map((x) => x.name === file.name ? { ...x, progress: pct } : x))
        })
        console.log('[DocuSage] upload success', { file: file.name, response: result.data })
        setUploads((u) => u.map((x) => x.name === file.name ? { ...x, done: true } : x))
        await qc.invalidateQueries({ queryKey: ['documents', activeWorkspaceId] })
        toast.success(`${file.name} queued for indexing`)
        setTimeout(() => {
          setUploads((u) => u.filter((x) => x.name !== file.name))
        }, 6000)
      } catch (err) {
        const message = getApiErrorMessage(err, 'Upload failed')
        console.error('[DocuSage] upload error', { file: file.name, message, err })
        setUploads((u) => u.map((x) =>
          x.name === file.name ? { ...x, error: true, errorMessage: message } : x
        ))
        toast.error(`${file.name}: ${message}`, { duration: 8000 })
        setTimeout(() => {
          setUploads((u) => u.filter((x) => x.name !== file.name))
        }, 15000)
      }
    }
  }, [activeWorkspaceId, qc])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'text/html': ['.html'],
    },
    multiple: true,
  })

  const docs = data?.documents ?? []
  const total = data?.total ?? 0
  const ready = docs.filter(d => d.status === 'completed').length

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-teal/10 shrink-0">
        <Database className="w-4 h-4 text-teal glow-text" />
        <h2 className="font-hud text-sm font-semibold text-bright flex-1 tracking-wide glow-text">Document Library</h2>

        {/* Stats */}
        <div className="flex items-center gap-1 font-mono text-xs text-dim">
          <span className="text-green">{ready}</span>
          <span>/</span>
          <span>{total}</span>
          <span className="ml-0.5">indexed</span>
        </div>

        <button onClick={onClose} className="text-dim hover:text-base transition-colors ml-2">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Drop zone */}
      {activeWorkspaceId && (
        <div
          {...getRootProps()}
          className={clsx(
            'mx-4 mt-3 mb-2 border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all',
            isDragActive
              ? 'border-teal bg-teal/5 scale-[0.99]'
              : 'border-border hover:border-teal/40 hover:bg-surface/50'
          )}
        >
          <input {...getInputProps()} />
          <Upload className={clsx('w-5 h-5 mx-auto mb-2', isDragActive ? 'text-teal' : 'text-dim')} />
          <p className="text-xs text-subtle">
            {isDragActive ? 'Drop to upload' : 'Drag files here or click to browse'}
          </p>
          <p className="text-2xs text-dim mt-1 font-mono">PDF · DOCX · TXT · MD · HTML</p>
        </div>
      )}

      {/* Indexing guidance banner */}
      {docs.some(d => IN_FLIGHT_STATUSES.has(d.status)) && (
        <div className="mx-4 mt-2 mb-1 px-3 py-2 bg-amber/5 border border-amber/20 rounded-lg">
          <p className="text-xs text-amber leading-relaxed">
            Documents are indexing — they will appear in answers once status shows "Ready"
          </p>
        </div>
      )}

      {/* Upload progress */}
      <UploadProgress items={uploads} />

      {/* Documents list */}
      <div className="flex-1 overflow-y-auto">
        {!activeWorkspaceId ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <Database className="w-8 h-8 text-dim mb-3" />
            <p className="text-sm text-subtle">Select a workspace to see documents</p>
          </div>
        ) : isLoading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="w-5 h-5 text-teal animate-spin" />
          </div>
        ) : docs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-center px-8">
            <FileText className="w-8 h-8 text-dim mb-3" />
            <p className="text-sm text-subtle">No documents yet</p>
            <p className="text-xs text-dim mt-1">Upload files above to start indexing</p>
          </div>
        ) : (
          docs.map((doc) => (
            <DocRow
              key={doc.id}
              doc={doc}
              onDelete={() => setConfirmDeleteId(doc.id)}
            />
          ))
        )}
      </div>

      {confirmDeleteId && (
        <ConfirmDialog
          title="Delete document?"
          description="This will permanently remove the document and all its indexed chunks."
          confirmLabel="Delete"
          danger
          onConfirm={() => { deleteMut.mutate(confirmDeleteId); setConfirmDeleteId(null) }}
          onCancel={() => setConfirmDeleteId(null)}
        />
      )}
    </div>
  )
}
