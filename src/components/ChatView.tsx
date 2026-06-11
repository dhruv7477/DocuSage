import { useState, useRef, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Send, Bot, User, Loader2, Sparkles, ChevronDown, AlertTriangle, Info, X } from 'lucide-react'
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
import { CitationLens } from './CitationLens'
import { SourcePanel } from './SourcePanel'

// ── Typewriter-style streaming cursor ────────────────────────────
function StreamingContent({ content }: { content: string }) {
  return (
    <span>
      {content}
      <span className="inline-block w-0.5 h-3.5 bg-teal ml-0.5 align-middle animate-blink" />
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

// ── Citation chip + helpers ───────────────────────────────────────

// Highlight-and-scroll a source badge when its [n] chip is clicked.
function focusSource(messageId: string, index: number) {
  if (typeof document === 'undefined') return
  const el = document.getElementById(`source-${messageId}-${index}`)
  if (!el) return
  el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  el.classList.add('ring-2', 'ring-teal/60')
  window.setTimeout(() => el.classList.remove('ring-2', 'ring-teal/60'), 1400)
}

// Split assistant text on [n] / [n][m] markers and render each marker as
// a clickable chip. Out-of-range markers (no matching source) render as
// dim text — better than a broken link.
function renderWithCitations(
  content: string,
  sources: CitedSource[],
  messageId: string,
  userQuery?: string,
) {
  if (!sources.length) return <>{content}</>

  const parts: (string | { idx: number })[] = []
  const re = /\[(\d+)\]/g
  let lastEnd = 0
  let match: RegExpExecArray | null
  while ((match = re.exec(content)) !== null) {
    if (match.index > lastEnd) parts.push(content.slice(lastEnd, match.index))
    parts.push({ idx: parseInt(match[1], 10) - 1 })
    lastEnd = match.index + match[0].length
  }
  if (lastEnd < content.length) parts.push(content.slice(lastEnd))

  return (
    <>
      {parts.map((p, i) => {
        if (typeof p === 'string') return <span key={i}>{p}</span>
        const isValid = p.idx >= 0 && p.idx < sources.length
        if (!isValid) {
          return <span key={i} className="font-mono text-dim">[{p.idx + 1}]</span>
        }
        const chip = (
          <button
            onClick={() => focusSource(messageId, p.idx)}
            className="inline-flex align-middle items-center font-mono text-2xs font-bold text-teal/90 bg-teal/8 hover:bg-teal/15 hover:text-teal border border-teal/20 hover:border-teal/40 rounded px-1 mx-0.5 transition-colors"
            title={`Source ${p.idx + 1}: ${sources[p.idx].source}`}
          >
            [{p.idx + 1}]
          </button>
        )
        return (
          <CitationLens key={i} source={sources[p.idx]} query={userQuery}>
            {chip}
          </CitationLens>
        )
      })}
    </>
  )
}

// ── Source citation badge ─────────────────────────────────────────
function SourceBadge({
  source,
  index,
  messageId,
}: {
  source: CitedSource
  index: number
  messageId: string
}) {
  const heading = source.heading_path && source.heading_path.length
    ? source.heading_path.join(' > ')
    : null
  return (
    <div
      id={`source-${messageId}-${index}`}
      className="flex flex-col gap-1 px-3 py-2 bg-surface border border-border rounded-lg text-xs group hover:border-teal/30 transition-all cursor-default"
    >
      <div className="flex items-center gap-2">
        <span className="font-mono text-teal/80 font-bold">[{index + 1}]</span>
        <span className="text-ghost truncate flex-1">{source.source}</span>
        {source.page != null && (
          <span className="text-dim font-mono">p.{source.page}</span>
        )}
        <span className="text-dim font-mono ml-auto">
          {(source.score * 100).toFixed(0)}%
        </span>
      </div>
      {heading && (
        <p className="font-mono text-2xs text-dim truncate pl-6" title={heading}>
          {heading}
        </p>
      )}
    </div>
  )
}

// ── Message bubble ────────────────────────────────────────────────
function MessageBubble({ msg, userQuery }: { msg: Message; userQuery?: string }) {
  const isUser = msg.role === 'user'
  const sources: CitedSource[] = msg.sources_json ?? []

  return (
    <div className={clsx('flex gap-3 animate-fade-up', isUser && 'flex-row-reverse')}>
      {/* Avatar */}
      <div className={clsx(
        'w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5',
        isUser
          ? 'bg-amber/10 border border-amber/20'
          : 'bg-teal/10 border border-teal/20'
      )}>
        {isUser
          ? <User className="w-3.5 h-3.5 text-amber" />
          : <Bot className="w-3.5 h-3.5 text-teal" />
        }
      </div>

      <div className={clsx('flex-1 max-w-2xl', isUser && 'flex flex-col items-end')}>
        {/* Content */}
        <div className={clsx(
          'rounded-xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap',
          isUser
            ? 'bg-amber/8 border border-amber/15 text-base'
            : 'bg-panel border border-border text-base'
        )}>
          {isUser ? msg.content : renderWithCitations(msg.content, sources, msg.id, userQuery)}
        </div>

        {/* Sources */}
        {sources.length > 0 && (
          <div className="mt-2 w-full space-y-1">
            <p className="font-mono text-2xs text-dim uppercase tracking-widest px-1 mb-1.5">Sources</p>
            {sources.map((s, i) => (
              <SourceBadge key={i} source={s} index={i} messageId={msg.id} />
            ))}
          </div>
        )}

        {/* Timestamp */}
        <p className="mt-1 font-mono text-2xs text-dim px-1">
          {formatDistanceToNow(new Date(msg.created_at), { addSuffix: true })}
        </p>
      </div>
    </div>
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
        <div className="w-16 h-16 rounded-2xl bg-teal/5 border border-teal/20 flex items-center justify-center">
          <Sparkles className="w-7 h-7 text-teal" />
        </div>
        <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-green border-2 border-void" />
      </div>
      <h2 className="font-display text-xl font-semibold text-bright mb-2">
        Ask about your documents
      </h2>
      <p className="text-sm text-subtle max-w-xs mb-8">
        The AI has read your workspace documents and is ready to answer questions with citations.
      </p>
      <div className="grid grid-cols-1 gap-2 w-full max-w-md">
        {starters.map((q) => (
          <button
            key={q}
            onClick={() => onSend(q)}
            className="text-left px-4 py-3 text-xs text-subtle border border-border rounded-lg hover:border-teal/30 hover:text-base hover:bg-teal/[0.03] transition-all"
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
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const qc = useQueryClient()
  const sse = useSSE()

  const { data: messages = [], isLoading } = useQuery({
    queryKey: ['messages', activeChatId],
    queryFn: () => chatApi.messages(activeChatId!).then((r) => r.data),
    enabled: !!activeChatId,
    refetchInterval: false,
  })

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

  // Scroll to bottom on new messages
  useEffect(() => {
    if (atBottom) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, sse.content, atBottom])

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80)
  }

  const sendMessage = async (q = input.trim()) => {
    if (!q || !activeChatId || sse.streaming || !hasReadyDoc) return
    setInput('')
    textareaRef.current?.focus()

    // Optimistically add user message
    const tempMsg: Message = {
      id: '__temp__',
      chat_id: activeChatId,
      role: 'user',
      content: q,
      token_count: null,
      sources_json: null,
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
        className="flex-1 overflow-y-auto"
      >
        {isLoading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="w-5 h-5 text-teal animate-spin" />
          </div>
        ) : messages.filter(m => m.id !== '__temp__').length === 0 && !sse.streaming ? (
          <EmptyChat onSend={sendMessage} />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
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
                <div className="w-7 h-7 rounded-lg bg-teal/10 border border-teal/20 flex items-center justify-center shrink-0 mt-0.5 animate-pulse-teal">
                  <Bot className="w-3.5 h-3.5 text-teal" />
                </div>
                <div className="flex-1 max-w-2xl">
                  <div className="rounded-xl px-4 py-3 text-sm leading-relaxed bg-panel border border-teal/20">
                    {sse.content
                      ? <StreamingContent content={sse.content} />
                      : <span className="flex items-center gap-2 text-dim">
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          {sse.stage === 'retrieving' ? 'Searching documents…'
                           : sse.stage === 'generating' ? 'Generating answer…'
                           : 'Thinking…'}
                        </span>
                    }
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
              <div className="space-y-1 pl-10">
                <p className="font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Sources</p>
                {sse.sources.map((s, i) => (
                  <SourceBadge key={i} source={s} index={i} messageId="streaming" />
                ))}
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Scroll to bottom button */}
      {!atBottom && (
        <button
          onClick={() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' })}
          className="absolute bottom-24 right-8 w-8 h-8 rounded-full bg-panel border border-border text-subtle hover:text-base shadow-panel flex items-center justify-center z-10 transition-colors"
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

      {/* Input area */}
      <div className="shrink-0 border-t border-border bg-surface/80 backdrop-blur-sm px-4 py-3">
        <div className="max-w-3xl mx-auto">
          <div className={clsx(
            'flex items-end gap-2 bg-panel rounded-xl border transition-all',
            sse.streaming ? 'border-teal/20' : 'border-border focus-within:border-teal/40',
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
              className="flex-1 bg-transparent resize-none text-sm text-base placeholder:text-dim p-3 outline-none max-h-40 leading-relaxed disabled:cursor-not-allowed"
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || composerDisabled}
              className={clsx(
                'mb-2.5 mr-2.5 w-8 h-8 rounded-lg flex items-center justify-center transition-all',
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
    <div className="w-16 h-16 rounded-2xl bg-surface border border-border flex items-center justify-center mx-auto">
      <Bot className="w-7 h-7 text-dim" />
    </div>
  )
}
