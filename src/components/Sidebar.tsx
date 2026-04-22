import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  MessageSquare, Plus, Zap, ChevronDown,
  Trash2, LogOut, Settings, Settings2, X, Check
} from 'lucide-react'
import { clsx } from 'clsx'
import { formatDistanceToNow } from 'date-fns'
import toast from 'react-hot-toast'
import { workspaces as wsApi, chat as chatApi } from '@/api'
import { useAuth } from '@/store/auth'
import { useApp } from '@/store/app'
import { Button } from '@/components/ui/Button'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import { SettingsModal } from '@/components/SettingsModal'
import { WorkspaceSettingsModal } from '@/components/WorkspaceSettingsModal'

// ── Section header ────────────────────────────────────────────────
function SectionHeader({ label, onAdd }: { label: string; onAdd?: () => void }) {
  return (
    <div className="flex items-center justify-between px-3 mb-1">
      <span className="font-mono text-2xs text-dim uppercase tracking-widest">{label}</span>
      {onAdd && (
        <button
          onClick={onAdd}
          className="w-4 h-4 flex items-center justify-center text-dim hover:text-teal transition-colors"
        >
          <Plus className="w-3 h-3" />
        </button>
      )}
    </div>
  )
}

// ── Workspace picker ──────────────────────────────────────────────
function WorkspacePicker() {
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [wsSettingsId, setWsSettingsId] = useState<string | null>(null)
  const { activeWorkspaceId, setWorkspace } = useApp()
  const qc = useQueryClient()

  const { data: wsData } = useQuery({
    queryKey: ['workspaces'],
    queryFn: () => wsApi.list().then((r) => r.data),
  })

  const createMut = useMutation({
    mutationFn: () => wsApi.create(newName.trim()),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['workspaces'] })
      setWorkspace(res.data.id)
      setCreating(false)
      setNewName('')
      setOpen(false)
    },
    onError: () => toast.error('Failed to create workspace'),
  })

  // Auto-select first workspace when none is active
  useEffect(() => {
    if (!activeWorkspaceId && wsData && wsData.length > 0) {
      setWorkspace(wsData[0].id)
    }
  }, [wsData, activeWorkspaceId, setWorkspace])

  const active = wsData?.find((w) => w.id === activeWorkspaceId)

  return (
    <div className="relative px-3 mb-4">
      <button
        onClick={() => setOpen((v) => !v)}
        className={clsx(
          'w-full flex items-center gap-2.5 h-9 px-3 rounded-lg transition-all',
          'bg-surface border border-border hover:border-teal/30',
          open && 'border-teal/40'
        )}
      >
        <div className="w-5 h-5 rounded bg-teal/10 border border-teal/20 flex items-center justify-center shrink-0">
          <Zap className="w-3 h-3 text-teal" />
        </div>
        <span className="flex-1 text-left text-sm text-base truncate font-medium">
          {active?.name ?? 'Select workspace'}
        </span>
        <ChevronDown className={clsx('w-3.5 h-3.5 text-dim transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute top-full left-3 right-3 mt-1 z-50 bg-panel border border-border rounded-lg shadow-panel overflow-hidden">
          {wsData?.map((ws) => (
            <div
              key={ws.id}
              className={clsx(
                'group flex items-center gap-2 px-3 py-2 text-sm transition-colors cursor-pointer',
                ws.id === activeWorkspaceId
                  ? 'bg-teal/10 text-teal'
                  : 'text-ghost hover:bg-surface hover:text-bright'
              )}
              onClick={() => { setWorkspace(ws.id); setOpen(false) }}
            >
              <div className="w-1.5 h-1.5 rounded-full bg-current opacity-50 shrink-0" />
              <span className="flex-1 truncate text-left">{ws.name}</span>
              {ws.id === activeWorkspaceId && <Check className="w-3 h-3 shrink-0" />}
              <button
                onClick={(e) => { e.stopPropagation(); setOpen(false); setWsSettingsId(ws.id) }}
                className="opacity-0 group-hover:opacity-100 text-dim hover:text-bright transition-all ml-1"
                title="Workspace settings"
              >
                <Settings2 className="w-3 h-3" />
              </button>
            </div>
          ))}

          <div className="border-t border-border p-2">
            {creating ? (
              <div className="flex items-center gap-1">
                <input
                  autoFocus
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && newName.trim()) createMut.mutate()
                    if (e.key === 'Escape') setCreating(false)
                  }}
                  placeholder="Workspace name…"
                  className="flex-1 bg-surface border border-border rounded px-2 h-7 text-xs text-base outline-none focus:border-teal/50 placeholder:text-dim"
                />
                <button
                  onClick={() => newName.trim() && createMut.mutate()}
                  className="h-7 w-7 flex items-center justify-center rounded bg-teal/10 text-teal hover:bg-teal/20"
                >
                  <Check className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setCreating(false)}
                  className="h-7 w-7 flex items-center justify-center rounded text-dim hover:text-ghost"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setCreating(true)}
                className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-dim hover:text-teal transition-colors rounded hover:bg-teal/5"
              >
                <Plus className="w-3 h-3" />
                New workspace
              </button>
            )}
          </div>
        </div>
      )}

      {wsSettingsId && (
        <WorkspaceSettingsModal
          workspaceId={wsSettingsId}
          onClose={() => setWsSettingsId(null)}
        />
      )}
    </div>
  )
}

// ── Chat list ─────────────────────────────────────────────────────
function ChatList() {
  const { activeWorkspaceId, activeChatId, setChat } = useApp()
  const qc = useQueryClient()
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')

  const { data: chats = [] } = useQuery({
    queryKey: ['chats', activeWorkspaceId],
    queryFn: () => chatApi.list(activeWorkspaceId!).then((r) => r.data),
    enabled: !!activeWorkspaceId,
  })

  const { toggleDocPanel, docPanelOpen } = useApp()

  const createMut = useMutation({
    mutationFn: () => chatApi.create(activeWorkspaceId!, undefined),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['chats', activeWorkspaceId] })
      setChat(res.data.id)
      if (!docPanelOpen) toggleDocPanel()
    },
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => chatApi.delete(id),
    onSuccess: (_, id) => {
      qc.invalidateQueries({ queryKey: ['chats', activeWorkspaceId] })
      if (activeChatId === id) setChat(null)
    },
  })

  const renameMut = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => chatApi.rename(id, title),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['chats', activeWorkspaceId] }),
  })

  const commitRename = (id: string) => {
    const trimmed = editTitle.trim()
    if (trimmed) renameMut.mutate({ id, title: trimmed })
    setEditingId(null)
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <SectionHeader
        label="Chats"
        onAdd={activeWorkspaceId ? () => createMut.mutate() : undefined}
      />
      <div className="space-y-0.5 px-2">
        {chats.length === 0 && (
          <p className="px-2 py-3 text-xs text-dim italic">
            {activeWorkspaceId ? 'No chats yet. Start one above.' : 'Select a workspace first.'}
          </p>
        )}
        {chats.map((c) => (
          <div
            key={c.id}
            className={clsx(
              'group flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer transition-all',
              c.id === activeChatId
                ? 'bg-teal/10 text-teal'
                : 'text-subtle hover:text-base hover:bg-surface'
            )}
            onClick={() => { if (editingId !== c.id) setChat(c.id) }}
          >
            <MessageSquare className="w-3.5 h-3.5 shrink-0 opacity-60" />
            {editingId === c.id ? (
              <input
                autoFocus
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onBlur={() => commitRename(c.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitRename(c.id)
                  if (e.key === 'Escape') setEditingId(null)
                }}
                onClick={(e) => e.stopPropagation()}
                className="flex-1 bg-surface border border-teal/40 rounded px-1.5 h-5 text-xs text-bright outline-none"
              />
            ) : (
              <span
                className="flex-1 text-xs truncate"
                onDoubleClick={(e) => {
                  e.stopPropagation()
                  setEditingId(c.id)
                  setEditTitle(c.title ?? '')
                }}
              >
                {c.title ?? 'New Chat'}
              </span>
            )}
            <button
              onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(c.id) }}
              className="opacity-0 group-hover:opacity-100 text-dim hover:text-red transition-all"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      {confirmDeleteId && (
        <ConfirmDialog
          title="Delete chat?"
          description="This will permanently delete the chat and all its messages."
          confirmLabel="Delete"
          danger
          onConfirm={() => { deleteMut.mutate(confirmDeleteId); setConfirmDeleteId(null) }}
          onCancel={() => setConfirmDeleteId(null)}
        />
      )}
    </div>
  )
}

// ── Main sidebar ──────────────────────────────────────────────────
export function Sidebar() {
  const { user, logout } = useAuth()
  const [settingsOpen, setSettingsOpen] = useState(false)

  return (
    <aside className="w-56 shrink-0 flex flex-col h-full bg-surface border-r border-border">
      {/* Logo */}
      <div className="h-12 flex items-center gap-2.5 px-4 border-b border-border shrink-0">
        <div className="w-6 h-6 rounded bg-teal/10 border border-teal/30 flex items-center justify-center">
          <Zap className="w-3.5 h-3.5 text-teal" strokeWidth={2.5} />
        </div>
        <span className="font-display text-sm font-bold text-bright tracking-tight">DocuSage</span>
        <div className="ml-auto">
          <div className="status-dot bg-green animate-pulse-teal" />
        </div>
      </div>

      {/* Workspace picker */}
      <div className="pt-3">
        <WorkspacePicker />
      </div>

      {/* Chats */}
      <ChatList />

      {/* Bottom nav */}
      <div className="border-t border-border p-2 space-y-0.5 shrink-0">
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start gap-2.5 text-subtle"
          onClick={() => setSettingsOpen(true)}
        >
          <Settings className="w-3.5 h-3.5" />
          Settings
        </Button>

        {/* User */}
        <div className="flex items-center gap-2 px-2 py-1.5 mt-1 border-t border-border">
          <div className="w-6 h-6 rounded-full bg-teal/10 border border-teal/30 flex items-center justify-center shrink-0">
            <span className="font-mono text-2xs text-teal font-bold uppercase">
              {user?.full_name?.[0] ?? 'U'}
            </span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs text-bright font-medium truncate">{user?.full_name}</p>
            <p className="text-2xs text-dim truncate">{user?.role}</p>
          </div>
          <button
            onClick={logout}
            className="text-dim hover:text-red transition-colors"
            title="Sign out"
          >
            <LogOut className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
    </aside>
  )
}
