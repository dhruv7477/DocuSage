import { useState, useRef, useEffect, useCallback } from 'react'
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Send, Bot, User, Loader2, Sparkles, ChevronDown, AlertTriangle, Info, X, Maximize2, Eye } from 'lucide-react'
import { clsx } from 'clsx'
import { formatDistanceToNow } from 'date-fns'
import {
  chat as chatApi,
  documents as docsApi,
  IN_FLIGHT_STATUSES,
  type CitedSource,
  type Message,
} from '@/api'
import { useApp } from '@/store/app'
import { useSSE, type StreamWarning } from '@/hooks/useSSE'
import { useSourcePanel } from '@/store/source-panel'
import { usePageImage } from '@/hooks/usePageImage'
import { CitationLens } from './CitationLens'
import { citedSourceNumbers, highlightChunkId, normalizeCitations, splitCitations } from './citations'
import { SourcePanel } from './SourcePanel'
import { ThinkingStatus } from './ThinkingStatus'
import { MessagesSkeleton } from './Skeletons'

function isPdf(filename: string): boolean {
  return /\.pdf$/i.test(filename)
}

// ── Typewriter-style streaming cursor ────────────────────────────
function StreamingContent({ content }: { content: string }) {
  return (
    <span>
      {content}
      <span className="inline-block w-0.5 h-3.5 bg-teal ml-0.5 align-middle animate-blink shadow-teal" />
    </span>
  )
}

// ── Error / warning banners ───────────────────────────────────────
function ErrorBanner({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div className="flex items-start gap-3 px-4 py-3 mx-4 my-3 bg-red/8 border border-red/30 rounded-lg text-sm text-base">
      <AlertTriangle className="w-4 h-4 text-red shrink-0 mt-0.5" />
      <div className="flex-1">
        <p className="font-medium text-red">Couldn't generate an answer</p>
        <p className="mt-0.5 text-subtle">{message}</p>
      </div>
      <button
        onClick={onDismiss}
        className="text-dim hover:text-base transition-colors shrink-0"
        aria-label="Dismiss"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

function NoticeBanner({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      role="status"
      className="flex items-start gap-3 px-4 py-3 mx-4 my-3 bg-teal/8 border border-teal/30 rounded-lg text-sm text-base"
    >
      <Info className="w-4 h-4 text-teal shrink-0 mt-0.5" />
      <p className="flex-1 text-subtle">{message}</p>
      <button
        onClick={onDismiss}
        className="text-dim hover:text-base transition-colors shrink-0"
        aria-label="Dismiss"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

function WarningBanner({ warning }: { warning: StreamWarning }) {
  const label =
    warning.code === 'low_confidence' ? 'Low confidence'
    : warning.code === 'no_documents' ? 'No documents indexed'
    : warning.code === 'partial_context' ? 'Partial match'
    : 'Notice'
  return (
    <div className="flex items-start gap-2 px-3 py-2 mt-2 bg-amber/8 border border-amber/25 rounded-lg text-xs">
      <Info className="w-3.5 h-3.5 text-amber shrink-0 mt-0.5" />
      <div className="flex-1">
        <span className="font-mono text-2xs uppercase tracking-widest text-amber/90 mr-2">{label}</span>
        <span className="text-subtle">{warning.message}</span>
      </div>
    </div>
  )
}

// An answer that cites none of the documents is general knowledge: it is
// labelled so, and has no sources.
function NotFromDocumentsLabel() {
  return (
    <p className="mt-1.5 px-1 inline-flex items-center gap-1.5 font-mono text-2xs uppercase tracking-widest text-amber/90">
      <Info className="w-3 h-3" aria-hidden="true" />
      Not from your documents
    </p>
  )
}

// ── Citation helpers ──────────────────────────────────────────────

// While an answer streams its sources are not known yet, so its [n]
// markers are hidden (with the spacing and punctuation they leave behind);
// the saved answer then shows them as chips. The trailing-partial rules keep
// a half-streamed "[1" or "【1†p" from flickering while tokens arrive. The
// last rule only starts at the beginning of a whitespace run, so a long run
// that is not followed by punctuation is scanned once, not once per character.
export function stripCitations(content: string): string {
  return normalizeCitations(content)
    .replace(/\[\d+\]/g, '')
    .replace(/\[\d*$/, '')
    .replace(/【[^】\n]{0,40}$/, '')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/(?<!\s)\s+([.,;:!?])/g, '$1')
}

// ── Inline page-screenshot preview (issue #1: prominent, always-visible) ──
// Renders the cited PDF page with the passage highlighted, directly under
// the source badge. Clicking it opens the full magnifier (SourcePanel).
function SourcePreview({ source }: { source: CitedSource }) {
  const openSource = useSourcePanel((s) => s.openSource)
  const pageNum = source.page && source.page >= 1 ? source.page : 1
  // crop=true → just the highlighted paragraph, not the whole page.
  const { data, isLoading, isError } = usePageImage(source.document_id, pageNum, highlightChunkId(source), 2, true)

  if (isError) return null
  if (isLoading) {
    return <div className="mt-2 w-full h-24 rounded-md bg-teal/[0.03] border border-teal/10 animate-pulse" />
  }
  if (!data) return null

  return (
    <button
      onClick={() => openSource(source)}
      className="group/preview relative mt-2 block w-full overflow-hidden rounded-md border border-teal/15 hover:border-teal/45 transition-all hover:shadow-teal"
      title="Click to open the full page"
    >
      <img
        src={data.url}
        alt={`${source.source} — cited passage on page ${pageNum}`}
        className="w-full h-auto"
        loading="lazy"
      />
      {!data.highlightFound && (
        <span className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-void/80 border border-amber/30 text-amber/90 font-mono text-2xs">
          full page
        </span>
      )}
      {/* Magnify affordance */}
      <span className="absolute inset-0 bg-gradient-to-t from-void/80 via-transparent to-transparent opacity-0 group-hover/preview:opacity-100 transition-opacity" />
      <span className="absolute bottom-2 right-2 flex items-center gap-1 px-2 py-1 rounded bg-void/85 border border-teal/40 text-teal font-mono text-2xs opacity-0 group-hover/preview:opacity-100 transition-opacity">
        <Maximize2 className="w-3 h-3" /> open page
      </span>
    </button>
  )
}

// ── Citation chip ─────────────────────────────────────────────────
// An inline [n] for source n. Hover or click shows the passage (and a way
// to open the page) via the CitationLens popover.
function CitationChip({ n, total, source, query }: { n: number; total: number; source: CitedSource; query?: string }) {
  return (
    <CitationLens source={source} query={query} rank={n} total={total}>
      <button
        type="button"
        aria-label={`Source ${n}`}
        title={`Source ${n}: ${source.source}`}
        className="inline-flex align-baseline items-center font-mono text-2xs font-bold text-teal/90 bg-teal/[0.08] hover:bg-teal/15 hover:text-teal border border-teal/20 hover:border-teal/40 rounded px-1 mx-0.5 transition-colors"
      >
        [{n}]
      </button>
    </CitationLens>
  )
}

// An answer with its [n] markers as chips; markers with no source are dropped.
function AnswerText({ content, sources, query }: { content: string; sources: CitedSource[]; query?: string }) {
  return (
    <>
      {splitCitations(content, sources.length).map((part, i) =>
        'text' in part
          ? <span key={i}>{part.text}</span>
          : <CitationChip key={i} n={part.source} total={sources.length} source={sources[part.source - 1]} query={query} />,
      )}
    </>
  )
}

// ── Source carousel card ──────────────────────────────────────────
// `label` is [n] for a cited source, or #rank by relevance. `rank` is the
// source's position among the answer's passages: the reranker's score is not
// calibrated, so it is not shown as a "% match". `showPreview` renders the
// cropped paragraph screenshot (reserved for the first cards). Hover or click
// reveals the full passage via the CitationLens popover.
function SourceCard({
  source,
  label,
  rank,
  total,
  showPreview,
  query,
}: {
  source: CitedSource
  label: string
  rank: number
  total: number
  showPreview: boolean
  query?: string
}) {
  const heading = source.heading_path && source.heading_path.length
    ? source.heading_path.join(' > ')
    : null
  return (
    <div className="snap-start shrink-0 w-60">
      <CitationLens source={source} query={query} block rank={rank} total={total}>
        <div className="h-full hud-panel rounded-lg p-2.5 cursor-pointer hover:border-teal/40 hover:shadow-teal transition-all group">
          <div className="flex items-center gap-2 text-xs">
            <span className="font-mono text-teal font-bold glow-text shrink-0">{label}</span>
            <span className="text-subtle truncate flex-1" title={source.source}>{source.source}</span>
            <Eye className="w-3.5 h-3.5 text-dim group-hover:text-teal transition-colors shrink-0" />
          </div>
          <div className="flex items-center gap-2 mt-0.5 font-mono text-2xs text-dim">
            {source.page != null && <span>p.{source.page}</span>}
            <span className="ml-auto text-teal/70">rank {rank} of {total}</span>
          </div>
          {heading && (
            <p className="font-mono text-2xs text-dim truncate mt-1" title={heading}>{heading}</p>
          )}
          {showPreview && isPdf(source.source) && <SourcePreview source={source} />}
        </div>
      </CitationLens>
    </div>
  )
}

// ── Collapsible source carousel ───────────────────────────────────
// Hidden behind a "Check sources" toggle. When the answer cites sources, it
// lists exactly those, in the order first cited and under the same [n] as
// their chips. Otherwise it keeps only the most relevant source(s): a single
// source document → top 2; multiple documents → just the top passage. The
// sources arrive in relevance order (source n backs [n]), so rank is position.
// The first cards show their cropped paragraph screenshot.
const PREVIEW_CARDS = 2

function keptSources(
  sources: CitedSource[],
  cited: number[],
): { source: CitedSource; label: string; rank: number }[] {
  if (cited.length) return cited.map((n) => ({ source: sources[n - 1], label: `[${n}]`, rank: n }))
  const distinctDocs = new Set(sources.map((s) => s.document_id)).size
  const keepCount = distinctDocs === 1 ? 2 : 1
  return sources.slice(0, keepCount).map((source, i) => ({ source, label: `#${i + 1}`, rank: i + 1 }))
}

function SourcesDisclosure({
  sources,
  cited,
  query,
}: {
  sources: CitedSource[]
  cited: number[]
  query?: string
}) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null

  const kept = keptSources(sources, cited)

  return (
    <div className="mt-2 w-full min-w-0">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg hud-panel font-mono text-2xs uppercase tracking-widest text-teal/80 hover:text-teal hover:border-teal/40 hover:shadow-teal transition-all"
      >
        <ChevronDown className={clsx('w-3.5 h-3.5 transition-transform', open && 'rotate-180')} />
        Check {kept.length === 1 ? 'source' : 'sources'}
        <span className="px-1.5 py-0.5 rounded-full bg-teal/10 border border-teal/25 text-teal">
          {kept.length}
        </span>
      </button>
      {open && (
        <div className="mt-1.5 flex gap-2 overflow-x-auto pb-2 snap-x snap-mandatory animate-fade-up">
          {kept.map(({ source, label, rank }, i) => (
            <SourceCard
              key={`${source.document_id}-${source.chunk_id ?? i}`}
              source={source}
              label={label}
              rank={rank}
              total={sources.length}
              showPreview={i < PREVIEW_CARDS}
              query={query}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Message bubble ────────────────────────────────────────────────
function MessageBubble({ msg, userQuery }: { msg: Message; userQuery?: string }) {
  const isUser = msg.role === 'user'
  const sources: CitedSource[] = msg.sources ?? []
  const cited = isUser ? [] : citedSourceNumbers(msg.content, sources.length)

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className={clsx('flex gap-3', isUser && 'flex-row-reverse')}
    >
      {/* Avatar */}
      <div className={clsx(
        'w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5',
        isUser
          ? 'bg-plasma/10 border border-plasma/30'
          : 'bg-teal/10 border border-teal/30 shadow-teal'
      )}>
        {isUser
          ? <User className="w-3.5 h-3.5 text-plasma" />
          : <Bot className="w-3.5 h-3.5 text-teal" />
        }
      </div>

      <div className={clsx('flex-1 min-w-0 max-w-2xl', isUser && 'flex flex-col items-end')}>
        {/* Content */}
        <div className={clsx(
          'rounded-xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap break-words',
          isUser
            ? 'bg-plasma/[0.07] border border-plasma/25 text-bright'
            : 'hud-panel text-bright'
        )}>
          {isUser ? msg.content : <AnswerText content={msg.content} sources={sources} query={userQuery} />}
        </div>

        {msg.label === 'not_from_documents' && <NotFromDocumentsLabel />}

        {/* Sources (collapsed dropdown; RAG answers only) */}
        {sources.length > 0 && <SourcesDisclosure sources={sources} cited={cited} query={userQuery} />}

        {/* Timestamp */}
        <p className="mt-1 font-mono text-2xs text-dim px-1">
          {formatDistanceToNow(new Date(msg.created_at), { addSuffix: true })}
        </p>
      </div>
    </motion.div>
  )
}

// ── Empty state ───────────────────────────────────────────────────
function EmptyChat({ onSend }: { onSend: (q: string) => void }) {
  const starters = [
    'Summarize the key findings in the uploaded documents',
    'What are the main topics covered across all documents?',
    'Find all mentions of action items or next steps',
    'Create a brief executive summary from the documents',
  ]

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
      <div className="relative mb-6">
        <div className="w-16 h-16 rounded-2xl bg-teal/5 border border-teal/25 flex items-center justify-center shadow-teal hud-corners relative">
          <Sparkles className="w-7 h-7 text-teal glow-text" />
        </div>
        <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-green border-2 border-void animate-glow-pulse" />
      </div>
      <h2 className="font-hud text-xl font-semibold text-bright mb-2 tracking-wide glow-text">
        Query your documents
      </h2>
      <p className="text-sm text-subtle max-w-xs mb-8">
        The assistant has scanned your workspace and is ready to answer with cited, source-grounded intelligence.
      </p>
      <div className="grid grid-cols-1 gap-2 w-full max-w-md">
        {starters.map((q) => (
          <button
            key={q}
            onClick={() => onSend(q)}
            className="text-left px-4 py-3 text-xs text-subtle hud-panel rounded-lg hover:border-teal/40 hover:text-bright hover:shadow-teal transition-all"
          >
            {q}
          </button>
        ))}
      </div>
      <p className="mt-6 font-mono text-2xs text-dim max-w-xs">
        Documents must be fully indexed (status "Ready") before they appear in answers.
      </p>
    </div>
  )
}

// ── Main Chat view ────────────────────────────────────────────────
export function ChatView() {
  const { activeChatId, activeWorkspaceId } = useApp()
  const [input, setInput] = useState('')
  const [atBottom, setAtBottom] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const qc = useQueryClient()
  const sse = useSSE()

  const { data: messages = [], isLoading } = useQuery({
    queryKey: ['messages', activeChatId],
    queryFn: () => chatApi.messages(activeChatId!).then((r) => r.data),
    enabled: !!activeChatId && activeChatId !== '__optimistic__',
    refetchInterval: false,
    staleTime: 30_000,
    placeholderData: keepPreviousData,
  })

  // Kill any in-flight stream when the user switches chats — without
  // this, tokens from chat A keep painting into chat B's view.
  useEffect(() => {
    sse.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeChatId])

  // On stream error, the backend has already committed the user message
  // (it commits before generation). The 'done' path invalidates messages;
  // the error path doesn't, so the optimistic __temp__ row would linger and
  // then vanish on the next refetch. Refetch here so the persisted user
  // message replaces the temp row and the failed turn stays visible.
  useEffect(() => {
    if ((!sse.error && !sse.notice) || !activeChatId) return
    qc.invalidateQueries({ queryKey: ['messages', activeChatId] })
  }, [sse.error, sse.notice, activeChatId, qc])

  // Fetch workspace documents to enforce the "at least one Ready doc"
  // guard. Poll while anything is still indexing so the composer
  // unlocks automatically once a document finishes.
  const { data: docsData } = useQuery({
    queryKey: ['documents', activeWorkspaceId],
    queryFn: () => docsApi.list(activeWorkspaceId!).then((r) => r.data),
    enabled: !!activeWorkspaceId,
    refetchInterval: (query) => {
      const docs = query.state.data?.documents ?? []
      const inFlight = docs.some(d => IN_FLIGHT_STATUSES.has(d.status))
      return inFlight ? 4000 : false
    },
  })
  const documents = docsData?.documents ?? []
  const readyDocCount = documents.filter(d => d.status === 'completed').length
  const indexingCount = documents.filter(d => IN_FLIGHT_STATUSES.has(d.status)).length
  const hasReadyDoc = readyDocCount > 0
  const composerDisabled = sse.streaming || !hasReadyDoc

  // Container-scoped scroll. scrollIntoView() walks up and scrolls *every*
  // scrollable ancestor — including document.body — to reveal the target.
  // With overflow:hidden on the body that drag is unrecoverable: the whole
  // shell ends up stranded above the viewport. Writing scrollTop on the
  // list element itself can only ever move the list.
  const scrollToBottom = useCallback((behavior: ScrollBehavior) => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior })
  }, [])

  // Scroll to bottom on new messages. rAF-throttled; instant during
  // streaming (smooth-scroll jitters at 10-15 tokens/s), smooth otherwise.
  useEffect(() => {
    if (!atBottom) return
    const id = requestAnimationFrame(() => {
      scrollToBottom(sse.streaming ? 'auto' : 'smooth')
    })
    return () => cancelAnimationFrame(id)
  }, [messages, sse.content, atBottom, sse.streaming, scrollToBottom])

  // Hard-jump to the end when opening/switching chats or when the real
  // data replaces placeholderData. keepPreviousData keeps the previous
  // thread's scroll geometry; without this the view can be stranded
  // past the end of a shorter thread (lone timestamp at the top,
  // emptiness below).
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const id = requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight
      setAtBottom(true)
    })
    return () => cancelAnimationFrame(id)
  }, [activeChatId, isLoading])

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80)
  }

  const sendMessage = (q = input.trim()) => {
    // '__optimistic__' is the placeholder id while chat creation is in
    // flight — block sends until the real id arrives.
    if (!q || !activeChatId || activeChatId === '__optimistic__' || sse.streaming || !hasReadyDoc) return
    setInput('')
    textareaRef.current?.focus()

    // Optimistically add user message
    const tempMsg: Message = {
      id: '__temp__',
      chat_id: activeChatId,
      role: 'user',
      content: q,
      token_count: null,
      sources: [],
      created_at: new Date().toISOString(),
    }
    qc.setQueryData<Message[]>(['messages', activeChatId], (old = []) => [...old, tempMsg])

    sse.start(chatApi.streamUrl(activeChatId, q), () => {
      qc.invalidateQueries({ queryKey: ['messages', activeChatId] })
      qc.invalidateQueries({ queryKey: ['chats', activeWorkspaceId] })
      sse.reset()
    })
  }

  const autoResize = () => {
    const ta = textareaRef.current
    if (ta) { ta.style.height = 'auto'; ta.style.height = `${ta.scrollHeight}px` }
  }

  if (!activeChatId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <MessageSquarePlaceholder />
          <p className="mt-4 text-sm text-subtle">Select or create a chat from the sidebar</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      <SourcePanel />
      {/* Messages */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 min-h-0 overflow-y-auto overscroll-contain"
      >
        {isLoading ? (
          <MessagesSkeleton />
        ) : messages.filter(m => m.id !== '__temp__').length === 0 && !sse.streaming ? (
          <EmptyChat onSend={sendMessage} />
        ) : (
          <div className="max-w-3xl mx-auto px-3 sm:px-4 py-6 pt-14 space-y-6">
            {(() => {
              const visible = messages.filter(m => m.role !== 'system')
              return visible.map((msg, idx) => {
                let userQuery: string | undefined
                if (msg.role === 'assistant') {
                  for (let j = idx - 1; j >= 0; j--) {
                    if (visible[j].role === 'user') {
                      userQuery = visible[j].content
                      break
                    }
                  }
                }
                return <MessageBubble key={msg.id} msg={msg} userQuery={userQuery} />
              })
            })()}

            {/* Streaming response */}
            {sse.streaming && (
              <div className="flex gap-3 animate-fade-up">
                <div className="w-7 h-7 rounded-lg bg-teal/10 border border-teal/30 flex items-center justify-center shrink-0 mt-0.5 animate-pulse-teal">
                  <Bot className="w-3.5 h-3.5 text-teal" />
                </div>
                <div className="flex-1 min-w-0 max-w-2xl">
                  <div className="rounded-xl px-4 py-3 text-sm leading-relaxed hud-panel break-words">
                    {sse.content
                      ? <StreamingContent content={stripCitations(sse.content)} />
                      : <ThinkingStatus stage={sse.stage} startedAt={sse.startedAt} />
                    }
                    {/* Post-generation validation runs after the last token —
                        keep the status line visible below the answer. */}
                    {sse.content && sse.stage === 'validating' && (
                      <div className="mt-2 pt-2 border-t border-border/50">
                        <ThinkingStatus stage={sse.stage} startedAt={sse.startedAt} />
                      </div>
                    )}
                  </div>
                  {sse.warnings.length > 0 && (
                    <div className="mt-1">
                      {sse.warnings.map((w, i) => <WarningBanner key={i} warning={w} />)}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Warnings after streaming completes (no-doc / abstention) */}
            {!sse.streaming && sse.warnings.length > 0 && (
              <div className="pl-10 space-y-1">
                {sse.warnings.map((w, i) => <WarningBanner key={i} warning={w} />)}
              </div>
            )}

            {/* Sources after streaming (before the persisted message arrives) */}
            {!sse.streaming && sse.sources.length > 0 && (
              <div className="pl-10">
                <SourcesDisclosure
                  sources={sse.sources}
                  cited={citedSourceNumbers(sse.content, sse.sources.length)}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Scroll to bottom button */}
      {!atBottom && (
        <button
          onClick={() => scrollToBottom('smooth')}
          aria-label="Scroll to latest"
          className="absolute bottom-28 right-4 sm:right-8 w-8 h-8 rounded-full hud-panel text-subtle hover:text-teal hover:shadow-teal flex items-center justify-center z-10 transition-all"
        >
          <ChevronDown className="w-4 h-4" />
        </button>
      )}

      {/* Error banner (rendered above the composer so it's always visible) */}
      {sse.error && (
        <div className="shrink-0">
          <ErrorBanner message={sse.error} onDismiss={sse.dismissError} />
        </div>
      )}
      {sse.notice && (
        <div className="shrink-0">
          <NoticeBanner message={sse.notice} onDismiss={sse.dismissNotice} />
        </div>
      )}

      {/* Input area — HUD console */}
      <div className="shrink-0 border-t border-teal/10 bg-surface/60 backdrop-blur-xl px-3 sm:px-4 py-3">
        <div className="max-w-3xl mx-auto">
          <div className={clsx(
            'relative flex items-end gap-2 hud-panel hud-corners rounded-xl transition-all',
            sse.streaming ? 'border-teal/30 shadow-teal' : 'focus-within:border-teal/50 focus-within:shadow-teal',
            !hasReadyDoc && 'opacity-60'
          )}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => { setInput(e.target.value); autoResize() }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              placeholder={
                sse.streaming
                  ? 'Generating…'
                  : !hasReadyDoc
                    ? indexingCount > 0
                      ? `Waiting for ${indexingCount} document${indexingCount === 1 ? '' : 's'} to finish indexing…`
                      : 'Upload a document before asking questions'
                    : 'Ask anything about your documents…'
              }
              disabled={composerDisabled}
              rows={1}
              className="flex-1 min-w-0 bg-transparent resize-none text-sm text-bright placeholder:text-dim p-3 outline-none max-h-40 leading-relaxed disabled:cursor-not-allowed"
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || composerDisabled}
              aria-label="Send message"
              className={clsx(
                'mb-2.5 mr-2.5 w-8 h-8 rounded-lg flex items-center justify-center transition-all shrink-0',
                input.trim() && !composerDisabled
                  ? 'bg-teal text-void hover:bg-teal/90 shadow-teal'
                  : 'bg-surface text-dim cursor-not-allowed'
              )}
            >
              {sse.streaming
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <Send className="w-4 h-4" />
              }
            </button>
          </div>
          <p className="mt-1.5 text-center font-mono text-2xs text-dim">
            {!hasReadyDoc && indexingCount > 0
              ? <>indexing {indexingCount} document{indexingCount === 1 ? '' : 's'} · the composer unlocks once at least one is Ready</>
              : !hasReadyDoc
                ? <>upload a document to start · answers come strictly from your indexed files</>
                : <>⏎ send · ⇧⏎ newline · {readyDocCount} document{readyDocCount === 1 ? '' : 's'} indexed · answers include source citations</>
            }
          </p>
        </div>
      </div>
    </div>
  )
}

// ── No chat selected placeholder ──────────────────────────────────
function MessageSquarePlaceholder() {
  return (
    <div className="w-16 h-16 rounded-2xl hud-panel hud-corners flex items-center justify-center mx-auto">
      <Bot className="w-7 h-7 text-dim" />
    </div>
  )
}
