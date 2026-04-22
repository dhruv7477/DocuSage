import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { Zap, Eye, EyeOff } from 'lucide-react'
import { GoogleLogin } from '@react-oauth/google'
import { auth } from '@/api'
import { useAuth } from '@/store/auth'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'

// ── Shared bg decoration ──────────────────────────────────────────
function AuthShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative min-h-screen bg-void overflow-hidden flex items-center justify-center">
      {/* Grid */}
      <div className="absolute inset-0 bg-grid opacity-100" />

      {/* Radial glow */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_-10%,rgba(0,212,200,0.08),transparent)]" />

      {/* Corner accents */}
      <div className="absolute top-0 left-0 w-48 h-48 border-l border-t border-teal/10" />
      <div className="absolute bottom-0 right-0 w-48 h-48 border-r border-b border-teal/10" />

      {/* Diagonal stripe */}
      <div className="absolute -top-32 -right-32 w-96 h-96 bg-teal/[0.03] rotate-45" />

      {/* Logo */}
      <div className="absolute top-8 left-8 flex items-center gap-2.5">
        <div className="w-7 h-7 rounded bg-teal/10 border border-teal/30 flex items-center justify-center">
          <Zap className="w-4 h-4 text-teal" strokeWidth={2.5} />
        </div>
        <span className="font-display text-base font-semibold text-bright tracking-tight">
          DocuSage
        </span>
      </div>

      {/* Version badge */}
      <div className="absolute top-8 right-8 font-mono text-2xs text-dim border border-border px-2 py-1 rounded">
        v1.0
      </div>

      <div className="relative z-10 w-full max-w-sm px-4">
        {children}
      </div>
    </div>
  )
}

// ── Shared Google handler ─────────────────────────────────────────
function useGoogleAuth() {
  const { setTokens, setUser } = useAuth()
  const navigate = useNavigate()

  return useMutation({
    mutationFn: (credential: string) => auth.googleLogin(credential),
    onSuccess: async (res) => {
      setTokens(res.data.access_token, res.data.refresh_token)
      const me = await auth.me()
      setUser(me.data)
      navigate('/')
    },
    onError: () => toast.error('Google sign-in failed'),
  })
}

// ── Login ─────────────────────────────────────────────────────────
export function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const { setTokens, setUser } = useAuth()
  const navigate = useNavigate()
  const googleMut = useGoogleAuth()

  const loginMut = useMutation({
    mutationFn: () => auth.login(email, password),
    onSuccess: async (res) => {
      setTokens(res.data.access_token, res.data.refresh_token)
      const me = await auth.me()
      setUser(me.data)
      navigate('/')
    },
    onError: () => toast.error('Invalid credentials'),
  })

  return (
    <AuthShell>
      <div className="animate-fade-up">
        {/* Header */}
        <div className="mb-8">
          <p className="font-mono text-xs text-teal mb-2 tracking-widest uppercase">
            // AUTHENTICATE
          </p>
          <h1 className="font-display text-3xl font-bold text-bright leading-tight">
            Welcome back
          </h1>
          <p className="mt-2 text-sm text-subtle">
            Sign in to your private knowledge base.
          </p>
        </div>

        {/* Form */}
        <div className="bg-panel border border-border rounded-xl p-6 shadow-panel space-y-4">
          <Input
            label="Email"
            type="email"
            placeholder="you@company.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && loginMut.mutate()}
            autoFocus
          />
          <div className="relative">
            <Input
              label="Password"
              type={showPw ? 'text' : 'password'}
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && loginMut.mutate()}
              className="pr-10"
            />
            <button
              type="button"
              onClick={() => setShowPw((v) => !v)}
              className="absolute right-3 bottom-2.5 text-dim hover:text-ghost transition-colors"
            >
              {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>

          <Button
            variant="primary"
            size="lg"
            className="w-full mt-2 font-display font-semibold tracking-wide"
            loading={loginMut.isPending}
            onClick={() => loginMut.mutate()}
          >
            Sign In
          </Button>

          <div className="relative flex items-center gap-3 py-1">
            <div className="flex-1 h-px bg-border" />
            <span className="text-xs text-dim font-mono">or</span>
            <div className="flex-1 h-px bg-border" />
          </div>

          <div className="flex justify-center">
            <GoogleLogin
              onSuccess={(res) => res.credential && googleMut.mutate(res.credential)}
              onError={() => toast.error('Google sign-in failed')}
              theme="filled_black"
              shape="rectangular"
              size="large"
              text="signin_with"
              width="100%"
            />
          </div>
        </div>

        <p className="mt-5 text-center text-sm text-subtle">
          No account?{' '}
          <Link to="/register" className="text-teal hover:text-teal/80 transition-colors">
            Create one
          </Link>
        </p>
      </div>
    </AuthShell>
  )
}

// ── Register ──────────────────────────────────────────────────────
export function RegisterPage() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const { setTokens, setUser } = useAuth()
  const navigate = useNavigate()
  const googleMut = useGoogleAuth()

  const registerMut = useMutation({
    mutationFn: () => auth.register(email, password, name),
    onSuccess: async () => {
      const loginRes = await auth.login(email, password)
      setTokens(loginRes.data.access_token, loginRes.data.refresh_token)
      const me = await auth.me()
      setUser(me.data)
      navigate('/')
      toast.success('Account created!')
    },
    onError: () => toast.error('Registration failed. Email may already exist.'),
  })

  return (
    <AuthShell>
      <div className="animate-fade-up">
        <div className="mb-8">
          <p className="font-mono text-xs text-teal mb-2 tracking-widest uppercase">
            // NEW_INSTANCE
          </p>
          <h1 className="font-display text-3xl font-bold text-bright">
            Create account
          </h1>
          <p className="mt-2 text-sm text-subtle">
            Your private AI knowledge assistant awaits.
          </p>
        </div>

        <div className="bg-panel border border-border rounded-xl p-6 shadow-panel space-y-4">
          <Input
            label="Full Name"
            placeholder="Ada Lovelace"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />
          <Input
            label="Email"
            type="email"
            placeholder="you@company.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <div className="relative">
            <Input
              label="Password"
              type={showPw ? 'text' : 'password'}
              placeholder="Min 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && registerMut.mutate()}
              className="pr-10"
            />
            <button
              type="button"
              onClick={() => setShowPw((v) => !v)}
              className="absolute right-3 bottom-2.5 text-dim hover:text-ghost transition-colors"
            >
              {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>

          <Button
            variant="primary"
            size="lg"
            className="w-full mt-2 font-display font-semibold tracking-wide"
            loading={registerMut.isPending}
            onClick={() => registerMut.mutate()}
          >
            Create Account
          </Button>

          <div className="relative flex items-center gap-3 py-1">
            <div className="flex-1 h-px bg-border" />
            <span className="text-xs text-dim font-mono">or</span>
            <div className="flex-1 h-px bg-border" />
          </div>

          <div className="flex justify-center">
            <GoogleLogin
              onSuccess={(res) => res.credential && googleMut.mutate(res.credential)}
              onError={() => toast.error('Google sign-in failed')}
              theme="filled_black"
              shape="rectangular"
              size="large"
              text="signup_with"
              width="100%"
            />
          </div>
        </div>

        <p className="mt-5 text-center text-sm text-subtle">
          Already have an account?{' '}
          <Link to="/login" className="text-teal hover:text-teal/80 transition-colors">
            Sign in
          </Link>
        </p>
      </div>
    </AuthShell>
  )
}
