import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { X, User, Shield, Info } from 'lucide-react'
import { clsx } from 'clsx'
import toast from 'react-hot-toast'
import { auth as authApi } from '@/api'
import { useAuth } from '@/store/auth'
import { Button } from '@/components/ui/Button'
import { ModalOverlay } from '@/components/ui/ModalOverlay'

type Tab = 'profile' | 'security' | 'about'

// ── Profile tab ───────────────────────────────────────────────────
function ProfileTab() {
  const { user } = useAuth()
  return (
    <div className="space-y-4">
      <div>
        <label className="block font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Full name</label>
        <div className="h-9 px-3 rounded-lg bg-surface border border-border text-sm text-bright flex items-center">
          {user?.full_name}
        </div>
      </div>
      <div>
        <label className="block font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Email</label>
        <div className="h-9 px-3 rounded-lg bg-surface border border-border text-sm text-bright flex items-center">
          {user?.email}
        </div>
      </div>
      <div>
        <label className="block font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Role</label>
        <div className="h-9 px-3 rounded-lg bg-surface border border-border text-sm text-bright flex items-center">
          {user?.role}
        </div>
      </div>
    </div>
  )
}

// ── Security tab ──────────────────────────────────────────────────
function SecurityTab() {
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')

  const changeMut = useMutation({
    mutationFn: () => authApi.changePassword(current, next),
    onSuccess: () => {
      toast.success('Password updated')
      setCurrent(''); setNext(''); setConfirm('')
    },
    onError: () => toast.error('Failed to change password'),
  })

  const canSubmit = current && next && next === confirm && next.length >= 8

  return (
    <form
      className="space-y-4"
      onSubmit={(e) => { e.preventDefault(); if (canSubmit) changeMut.mutate() }}
    >
      <div>
        <label className="block font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Current password</label>
        <input
          type="password"
          value={current}
          onChange={(e) => setCurrent(e.target.value)}
          className="w-full h-9 px-3 rounded-lg bg-surface border border-border text-sm text-bright outline-none focus:border-teal/50 placeholder:text-dim"
          placeholder="••••••••"
        />
      </div>
      <div>
        <label className="block font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">New password</label>
        <input
          type="password"
          value={next}
          onChange={(e) => setNext(e.target.value)}
          className="w-full h-9 px-3 rounded-lg bg-surface border border-border text-sm text-bright outline-none focus:border-teal/50 placeholder:text-dim"
          placeholder="Min. 8 characters"
        />
      </div>
      <div>
        <label className="block font-mono text-2xs text-dim uppercase tracking-widest mb-1.5">Confirm new password</label>
        <input
          type="password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          className={clsx(
            'w-full h-9 px-3 rounded-lg bg-surface border text-sm text-bright outline-none focus:border-teal/50 placeholder:text-dim',
            confirm && confirm !== next ? 'border-red/50' : 'border-border'
          )}
          placeholder="Repeat new password"
        />
        {confirm && confirm !== next && (
          <p className="mt-1 text-xs text-red">Passwords do not match</p>
        )}
      </div>
      <Button
        type="submit"
        variant="primary"
        size="sm"
        disabled={!canSubmit}
        loading={changeMut.isPending}
      >
        Update password
      </Button>
    </form>
  )
}

// ── About tab ─────────────────────────────────────────────────────
function AboutTab() {
  return (
    <div className="space-y-3 text-sm text-subtle">
      <div className="flex justify-between">
        <span className="text-dim font-mono text-xs">Application</span>
        <span className="text-bright font-medium">DocuSage</span>
      </div>
      <div className="flex justify-between">
        <span className="text-dim font-mono text-xs">Version</span>
        <span className="font-mono text-bright">1.0.0</span>
      </div>
      <div className="flex justify-between">
        <span className="text-dim font-mono text-xs">Stack</span>
        <span className="font-mono text-teal text-xs">FastAPI · React · Qdrant</span>
      </div>
    </div>
  )
}

// ── Main modal ─────────────────────────────────────────────────────
export function SettingsModal({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<Tab>('profile')

  const tabs: { id: Tab; label: string; icon: typeof User }[] = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'about', label: 'About', icon: Info },
  ]

  return (
    <ModalOverlay onClose={onClose}>
      <div className="relative bg-panel border border-border rounded-xl shadow-panel w-full max-w-md mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 className="font-display text-base font-semibold text-bright">Settings</h2>
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
                    ? 'bg-teal/10 text-teal'
                    : 'text-subtle hover:text-bright hover:bg-surface'
                )}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                {label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0 p-5 min-h-[260px]">
            {tab === 'profile' && <ProfileTab />}
            {tab === 'security' && <SecurityTab />}
            {tab === 'about' && <AboutTab />}
          </div>
        </div>
      </div>
    </ModalOverlay>
  )
}
