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
import { DemoNotice } from '@/components/DemoNotice'

// Google sign-in needs a client ID at build time (VITE_GOOGLE_CLIENT_ID).
// Without one the button would only ever fail, so it is not shown.
const GOOGLE_ENABLED = Boolean(import.meta.env.VITE_GOOGLE_CLIENT_ID)

function GoogleSection({
  text,
  onCredential,
}: {
  text: 'signin_with' | 'signup_with'
  onCredential: (credential: string) => void
}) {
  if (!GOOGLE_ENABLED) return null
  return (
    <>
      <div className="relative flex items-center gap-3 py-1">
        <div className="flex-1 h-px bg-border" />
        <span className="text-xs text-dim font-mono">or</span>
        <div className="flex-1 h-px bg-border" />
      </div>

      <div className="flex justify-center">
        <GoogleLogin
          onSuccess={(res) => res.credential && onCredential(res.credential)}
          onError={() => toast.error('Google sign-in failed')}
          theme="filled_black"
          shape="rectangular"
          size="large"
          text={text}
          width="100%"
        />
      </div>
    </>
  )
}

// ── Shared bg decoration ──────────────────────────────────────────
function AuthShell({ children }: { children: React.ReactNode }) {
  // The document itself never scrolls (index.html hides its overflow for
  // the app shell), so this page is its own scroll container: when the form
  // is taller than the window (the demo notice, a large font size, a small
  // screen), it scrolls; when it fits, it stays centred as before.
  return (
    <div className="h-[100dvh] overflow-y-auto">
      <div className="relative min-h-full bg-void overflow-hidden flex items-center justify-center">
        {/* Scrolling alignment grid */}
        <div className="absolute inset-0 bg-grid motion-safe:animate-grid-scroll opacity-70" />

        {/* Radial glow + depth */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_-10%,rgba(31,240,230,0.10),transparent)]" />
        <div className="absolute -bottom-1/3 -left-1/4 w-[60vw] h-[60vh] rounded-full bg-[radial-gradient(circle,rgba(157,107,255,0.06),transparent_60%)] blur-3xl motion-safe:animate-aurora-slow pointer-events-none" />

        {/* CRT scanlines + drifting sweep */}
        <div className="absolute inset-0 bg-scanlines opacity-50 mix-blend-screen pointer-events-none" />
        <div className="absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-teal/[0.07] to-transparent pointer-events-none motion-safe:animate-scanline" />

        {/* Corner accents */}
        <div className="absolute top-0 left-0 w-48 h-48 border-l border-t border-teal/15" />
        <div className="absolute bottom-0 right-0 w-48 h-48 border-r border-b border-teal/15" />

        {/* Vignette */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,transparent_55%,rgba(0,0,0,0.55))]" />

        {/* Logo */}
        <div className="absolute top-6 sm:top-8 left-6 sm:left-8 flex items-center gap-2.5">
          <div className="w-7 h-7 rounded bg-teal/10 border border-teal/30 flex items-center justify-center shadow-teal">
            <Zap className="w-4 h-4 text-teal" strokeWidth={2.5} />
          </div>
          <span className="font-hud text-base font-semibold text-bright tracking-wider glow-text">
            DocuSage
          </span>
        </div>

        {/* Version badge */}
        <div className="absolute top-6 sm:top-8 right-6 sm:right-8 font-mono text-2xs text-teal/70 border border-teal/20 px-2 py-1 rounded">
          v1.0
        </div>

        {/* Vertical padding keeps a tall form clear of the logo above it. */}
        <div className="relative z-10 w-full max-w-sm px-4 py-20">
          {children}
        </div>
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
            {'// AUTHENTICATE'}
          </p>
          <h1 className="font-hud text-3xl font-bold text-bright leading-tight tracking-wide glow-text">
            Welcome back
          </h1>
          <p className="mt-2 text-sm text-subtle">
            Sign in to your private knowledge base.
          </p>
        </div>

        {/* Form */}
        <div className="relative hud-panel hud-corners rounded-xl p-6 shadow-hud space-y-4">
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

          <GoogleSection text="signin_with" onCredential={(c) => googleMut.mutate(c)} />
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
            {'// NEW_INSTANCE'}
          </p>
          <h1 className="font-hud text-3xl font-bold text-bright tracking-wide glow-text">
            Create account
          </h1>
          <p className="mt-2 text-sm text-subtle">
            Your private AI knowledge assistant awaits.
          </p>
        </div>

        <DemoNotice />

        <div className="relative hud-panel hud-corners rounded-xl p-6 shadow-hud space-y-4">
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

          <GoogleSection text="signup_with" onCredential={(c) => googleMut.mutate(c)} />
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
