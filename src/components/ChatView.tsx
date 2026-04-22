import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Send, Bot, User, ExternalLink, Loader2, Sparkles, ChevronDown } from 'lucide-react'
import { clsx } from 'clsx'
import { formatDistanceToNow } from 'date-fns'
import toast from 'react-hot-toast'
import { chat as chatApi, type Message, type CitedSource } from '@/api'
import { useApp } from '@/store/app'
import { useSSE } from '@/hooks/useSSE'
import { Button } from '@/components/ui/Button'

// ── Typewriter-style streaming cursor ────────────────────────────
function StreamingContent({ content }: { content: string }) {
  return (
    <span>
      {content}
      <span className="inline-block w-0.5 h-3.5 bg-teal ml-0.5 align-middle animate-blink" />
    </span>
  )
}

// ── Source citation badge ─────────────────────────────────────────
function SourceBadge({ source, index }: { source: CitedSource; index: number }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-surface border border-border rounded-lg text-xs group hover:border-teal/30 transition-colors cursor-default">
      <span className="font-mono text-teal/80 font-bold">[{index + 1}]</span>
      <span className="text-ghost truncate flex-1">{source.source}</span>
      {source.page && <span className="text-dim font-mono">p.{source.page}</span>}
      <span className="text-dim font-mono ml-auto">{(source.score * 100).toFixed(0)}%</span>
    </div>
  )
}

// ── Message bubble ────────────────────────────────────────────────
function MessageBubble({ msg }: { msg: Message }) {
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
          'rounded-xl px-4 py-3 text-sm leading-relaxed',
          isUser
            ? 'bg-amber/8 border border-amber/15 text-base'
            : 'bg-panel border border-border text-base'
        )}>
          {msg.content}
        </div>

        {/* Sources */}
        {sources.length > 0 && (
          <div className="mt-2 w-full space-y-1">
            <p className="font-mono text-2xs text-dim uppercase tracking-widest px-1 mb-1.5">Sources</p>
            {sources.map((s, i) => <SourceBadge key={i} source={s} index={i} />)}
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
    if (!q || !activeChatId || sse.streaming) return
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
            {messages
              .filter(m => m.role !== 'system' && m.id !== '__temp__')
              .map((msg) => <MessageBubble key={msg.id} msg={msg} />)
            }

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
                          Thinking…
                        </span>
                    }
                  </div>
                </div>
              </div>
            )}

            {/* Sources after streaming */}
            {!sse.streaming && sse.sources.length > 0 && (
              <div className="space-y-1 pl-10">
                <p className="font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Sources</p>
                {sse.sources.map((s, i) => <SourceBadge key={i} source={s} index={i} />)}
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

      {/* Input area */}
      <div className="shrink-0 border-t border-border bg-surface/80 backdrop-blur-sm px-4 py-3">
        <div className="max-w-3xl mx-auto">
          <div className={clsx(
            'flex items-end gap-2 bg-panel rounded-xl border transition-all',
            sse.streaming ? 'border-teal/20' : 'border-border focus-within:border-teal/40'
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
              placeholder={sse.streaming ? 'Generating…' : 'Ask anything about your documents…'}
              disabled={sse.streaming}
              rows={1}
              className="flex-1 bg-transparent resize-none text-sm text-base placeholder:text-dim p-3 outline-none max-h-40 leading-relaxed"
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || sse.streaming}
              className={clsx(
                'mb-2.5 mr-2.5 w-8 h-8 rounded-lg flex items-center justify-center transition-all',
                input.trim() && !sse.streaming
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
            ⏎ send · ⇧⏎ newline · answers include source citations
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
