import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { X, Settings2, Users, Trash2, UserMinus, Loader2 } from 'lucide-react'
import { clsx } from 'clsx'
import toast from 'react-hot-toast'
import { workspaces as wsApi } from '@/api'
import { useApp } from '@/store/app'
import { useAuth } from '@/store/auth'
import { Button } from '@/components/ui/Button'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import { ModalOverlay } from '@/components/ui/ModalOverlay'

type Tab = 'general' | 'members' | 'danger'

// ── General tab ───────────────────────────────────────────────────
function GeneralTab({ workspaceId }: { workspaceId: string }) {
  const qc = useQueryClient()
  const { data } = useQuery({
    queryKey: ['workspace', workspaceId],
    queryFn: () => wsApi.get(workspaceId).then((r) => r.data),
  })
  const [name, setName] = useState(data?.name ?? '')

  const updateMut = useMutation({
    mutationFn: () => wsApi.update(workspaceId, { name: name.trim() }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workspaces'] })
      qc.invalidateQueries({ queryKey: ['workspace', workspaceId] })
      toast.success('Workspace renamed')
    },
    onError: () => toast.error('Failed to rename workspace'),
  })

  return (
    <div className="space-y-4">
      <div>
        <label className="block font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Workspace name</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full h-9 px-3 rounded-lg bg-surface border border-border text-sm text-bright outline-none focus:border-teal/50 placeholder:text-dim"
          placeholder="My Workspace"
        />
      </div>
      <Button
        variant="primary"
        size="sm"
        disabled={!name.trim() || name.trim() === data?.name}
        loading={updateMut.isPending}
        onClick={() => updateMut.mutate()}
      >
        Save changes
      </Button>
    </div>
  )
}

// ── Members tab ───────────────────────────────────────────────────
function MembersTab({ workspaceId }: { workspaceId: string }) {
  const { user } = useAuth()
  const qc = useQueryClient()
  const [email, setEmail] = useState('')
  const [removingId, setRemovingId] = useState<string | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ['workspace', workspaceId],
    queryFn: () => wsApi.get(workspaceId).then((r) => r.data),
  })

  const addMut = useMutation({
    mutationFn: () => wsApi.addMember(workspaceId, email.trim()),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workspace', workspaceId] })
      toast.success(`Invited ${email.trim()}`)
      setEmail('')
    },
    onError: () => toast.error('Failed to add member — check the email address'),
  })

  const removeMut = useMutation({
    mutationFn: (userId: string) => wsApi.removeMember(workspaceId, userId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workspace', workspaceId] })
      toast.success('Member removed')
    },
    onError: () => toast.error('Failed to remove member'),
  })

  if (isLoading) {
    return <div className="flex justify-center py-8"><Loader2 className="w-5 h-5 text-teal animate-spin" /></div>
  }

  return (
    <div className="space-y-4">
      {/* Invite */}
      <div>
        <label className="block font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Invite by email</label>
        <div className="flex gap-2">
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && email.trim()) addMut.mutate() }}
            placeholder="user@example.com"
            className="flex-1 min-w-0 h-9 px-3 rounded-lg bg-surface border border-border text-sm text-bright outline-none focus:border-teal/50 placeholder:text-dim"
          />
          <Button
            variant="primary"
            size="sm"
            disabled={!email.trim()}
            loading={addMut.isPending}
            onClick={() => addMut.mutate()}
          >
            Invite
          </Button>
        </div>
      </div>

      {/* Member list */}
      <div className="space-y-1.5">
        {data?.members.map((m) => (
          <div key={m.id} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-surface border border-border">
            <div className="w-7 h-7 rounded-full bg-teal/10 border border-teal/20 flex items-center justify-center shrink-0">
              <span className="font-mono text-2xs text-teal font-bold uppercase">{m.user_full_name[0]}</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-bright truncate">{m.user_full_name}</p>
              <p className="text-2xs text-dim truncate font-mono">{m.user_email}</p>
            </div>
            <span className={clsx(
              'font-mono text-2xs px-2 py-0.5 rounded border',
              m.role === 'admin' ? 'text-amber border-amber/30 bg-amber/5' : 'text-dim border-border bg-surface'
            )}>
              {m.role}
            </span>
            {m.user_id !== data?.owner_id && m.user_email !== user?.email && (
              <button
                onClick={() => setRemovingId(m.user_id)}
                className="text-dim hover:text-red transition-colors"
              >
                <UserMinus className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        ))}
      </div>

      {removingId && (
        <ConfirmDialog
          title="Remove member?"
          description="This member will lose access to the workspace and all its documents."
          confirmLabel="Remove"
          danger
          onConfirm={() => { removeMut.mutate(removingId); setRemovingId(null) }}
          onCancel={() => setRemovingId(null)}
        />
      )}
    </div>
  )
}

// ── Danger zone tab ───────────────────────────────────────────────
function DangerTab({ workspaceId, onDeleted }: { workspaceId: string; onDeleted: () => void }) {
  const qc = useQueryClient()
  const { setWorkspace } = useApp()
  const [confirming, setConfirming] = useState(false)

  const deleteMut = useMutation({
    mutationFn: () => wsApi.delete(workspaceId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workspaces'] })
      setWorkspace(null)
      toast.success('Workspace deleted')
      onDeleted()
    },
    onError: () => toast.error('Failed to delete workspace'),
  })

  return (
    <div className="space-y-4">
      <div className="p-4 rounded-lg bg-red/5 border border-red/20">
        <p className="text-sm font-medium text-red mb-1">Delete workspace</p>
        <p className="text-xs text-subtle mb-3 leading-relaxed">
          Permanently deletes this workspace, all documents, and chat history. This cannot be undone.
        </p>
        <Button variant="danger" size="sm" onClick={() => setConfirming(true)}>
          <Trash2 className="w-3.5 h-3.5" />
          Delete workspace
        </Button>
      </div>

      {confirming && (
        <ConfirmDialog
          title="Delete workspace?"
          description="All documents, chats, and indexed data will be permanently removed. This cannot be undone."
          confirmLabel="Delete workspace"
          danger
          onConfirm={() => deleteMut.mutate()}
          onCancel={() => setConfirming(false)}
        />
      )}
    </div>
  )
}

// ── Main modal ─────────────────────────────────────────────────────
export function WorkspaceSettingsModal({
  workspaceId,
  onClose,
}: {
  workspaceId: string
  onClose: () => void
}) {
  const [tab, setTab] = useState<Tab>('general')

  const tabs: { id: Tab; label: string; icon: typeof Settings2 }[] = [
    { id: 'general', label: 'General', icon: Settings2 },
    { id: 'members', label: 'Members', icon: Users },
    { id: 'danger', label: 'Danger zone', icon: Trash2 },
  ]

  return (
    <ModalOverlay onClose={onClose}>
      <div className="relative bg-panel border border-border rounded-xl shadow-panel w-full max-w-lg mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 className="font-display text-base font-semibold text-bright">Workspace settings</h2>
          <button onClick={onClose} className="text-dim hover:text-bright transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex">
          {/* Sidebar tabs */}
          <div className="w-36 border-r border-border p-2 space-y-0.5">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={clsx(
                  'w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors',
                  tab === id
                    ? id === 'danger' ? 'bg-red/10 text-red' : 'bg-teal/10 text-teal'
                    : id === 'danger' ? 'text-red/60 hover:text-red hover:bg-red/5' : 'text-subtle hover:text-bright hover:bg-surface'
                )}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                {label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0 p-5 min-h-[280px]">
            {tab === 'general' && <GeneralTab workspaceId={workspaceId} />}
            {tab === 'members' && <MembersTab workspaceId={workspaceId} />}
            {tab === 'danger' && <DangerTab workspaceId={workspaceId} onDeleted={onClose} />}
          </div>
        </div>
      </div>
    </ModalOverlay>
  )
}
