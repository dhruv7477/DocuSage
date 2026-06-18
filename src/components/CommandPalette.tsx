import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import { MessageSquare, Zap, Plus, PanelRight, Search } from 'lucide-react'
import { chat as chatApi, workspaces as wsApi, type Chat, type Workspace } from '@/api'
import { useApp } from '@/store/app'

interface PaletteItem {
  id: string
  label: string
  hint: string
  icon: React.ReactNode
  action: () => void
}

/** Case-insensitive subsequence match; lower score = better. */
export function fuzzyScore(needle: string, haystack: string): number | null {
  const n = needle.toLowerCase()
  const h = haystack.toLowerCase()
  if (!n) return 0
  const idx = h.indexOf(n)
  if (idx >= 0) return idx                       // substring → best, earlier better
  let hi = 0
  let gaps = 0
  for (const ch of n) {
    const found = h.indexOf(ch, hi)
    if (found === -1) return null
    gaps += found - hi
    hi = found + 1
  }
  return 100 + gaps                              // subsequence → ranked after substrings
}

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [highlighted, setHighlighted] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const qc = useQueryClient()
  const { activeWorkspaceId, setChat, setWorkspace, toggleDocPanel } = useApp()

  const close = useCallback(() => {
    setOpen(false)
    setQuery('')
    setHighlighted(0)
  }, [])

  // Global Ctrl/Cmd+K
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      }
      if (e.key === 'Escape') close()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [close])

  useEffect(() => {
    if (open) {
      inputRef.current?.focus()
      // Warm the caches so items appear instantly.
      qc.prefetchQuery({
        queryKey: ['workspaces'],
        queryFn: () => wsApi.list().then((r) => r.data),
        staleTime: 30_000,
      })
      if (activeWorkspaceId) {
        qc.prefetchQuery({
          queryKey: ['chats', activeWorkspaceId],
          queryFn: () => chatApi.list(activeWorkspaceId).then((r) => r.data),
          staleTime: 30_000,
        })
      }
    }
  }, [open, qc, activeWorkspaceId])

  const items = useMemo<PaletteItem[]>(() => {
    if (!open) return []
    const chats = qc.getQueryData<Chat[]>(['chats', activeWorkspaceId]) ?? []
    const workspaces = qc.getQueryData<Workspace[]>(['workspaces']) ?? []

    const chatItems: PaletteItem[] = chats
      .filter((c) => c.id !== '__optimistic__')
      .map((c) => ({
        id: `chat-${c.id}`,
        label: c.title ?? 'New Chat',
        hint: 'Chat',
        icon: <MessageSquare className="w-3.5 h-3.5 text-teal/70" />,
        action: () => { setChat(c.id); close() },
      }))

    const wsItems: PaletteItem[] = workspaces.map((w) => ({
      id: `ws-${w.id}`,
      label: w.name,
      hint: 'Workspace',
      icon: <Zap className="w-3.5 h-3.5 text-amber/70" />,
      action: () => { setWorkspace(w.id); close() },
    }))

    const actions: PaletteItem[] = [
      {
        id: 'action-toggle-docs',
        label: 'Toggle documents panel',
        hint: 'Action',
        icon: <PanelRight className="w-3.5 h-3.5 text-subtle" />,
        action: () => { toggleDocPanel(); close() },
      },
    ]
    if (activeWorkspaceId) {
      actions.unshift({
        id: 'action-new-chat',
        label: 'New chat',
        hint: 'Action',
        icon: <Plus className="w-3.5 h-3.5 text-subtle" />,
        action: () => {
          chatApi.create(activeWorkspaceId, undefined).then((res) => {
            qc.invalidateQueries({ queryKey: ['chats', activeWorkspaceId] })
            setChat(res.data.id)
          })
          close()
        },
      })
    }

    return [...actions, ...chatItems, ...wsItems]
  }, [open, qc, activeWorkspaceId, setChat, setWorkspace, toggleDocPanel, close])

  const filtered = useMemo(() => {
    if (!query.trim()) return items
    return items
      .map((item) => ({ item, score: fuzzyScore(query.trim(), item.label) }))
      .filter((x): x is { item: PaletteItem; score: number } => x.score !== null)
      .sort((a, b) => a.score - b.score)
      .map((x) => x.item)
  }, [items, query])

  useEffect(() => setHighlighted(0), [query])

  const onInputKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlighted((i) => Math.min(filtered.length - 1, i + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlighted((i) => Math.max(0, i - 1))
    } else if (e.key === 'Enter' && filtered[highlighted]) {
      filtered[highlighted].action()
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            onClick={close}
            className="fixed inset-0 z-[60] bg-black/50"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: -8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -8 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            role="dialog"
            aria-label="Command palette"
            className="fixed z-[61] left-1/2 top-[18vh] -translate-x-1/2 w-[min(560px,92vw)] bg-panel/90 backdrop-blur-2xl border border-border rounded-xl shadow-panel overflow-hidden"
          >
            <div className="flex items-center gap-2 px-4 py-3 border-b border-border/60">
              <Search className="w-4 h-4 text-dim shrink-0" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onInputKey}
                placeholder="Jump to a chat, switch workspace, run an action…"
                className="flex-1 bg-transparent text-sm text-bright placeholder:text-dim outline-none"
              />
              <kbd className="font-mono text-2xs text-dim border border-border rounded px-1.5 py-0.5">esc</kbd>
            </div>
            <div className="max-h-[40vh] overflow-y-auto py-1.5">
              {filtered.length === 0 && (
                <p className="px-4 py-6 text-center text-xs text-dim italic">Nothing matches.</p>
              )}
              {filtered.map((item, i) => (
                <button
                  key={item.id}
                  onClick={item.action}
                  onMouseEnter={() => setHighlighted(i)}
                  className={`w-full flex items-center gap-3 px-4 py-2 text-left text-sm transition-colors ${
                    i === highlighted ? 'bg-teal/10 text-bright' : 'text-subtle'
                  }`}
                >
                  {item.icon}
                  <span className="flex-1 truncate">{item.label}</span>
                  <span className="font-mono text-2xs text-dim">{item.hint}</span>
                </button>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
