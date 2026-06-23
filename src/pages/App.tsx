import { useQuery } from '@tanstack/react-query'
import { PanelLeftClose, PanelLeft, FileText } from 'lucide-react'
import { useApp } from '@/store/app'
import { Sidebar } from '@/components/Sidebar'
import { ChatView } from '@/components/ChatView'
import { DocumentsPanel } from '@/components/DocumentsPanel'
import { CommandPalette } from '@/components/CommandPalette'
import { documents as docsApi } from '@/api'

// ── Decorative HUD corner reticle (purely cosmetic) ─────────────────
function CornerReticle({ className }: { className?: string }) {
  return (
    <div className={`absolute w-16 h-16 pointer-events-none ${className ?? ''}`}>
      <div className="absolute inset-0 border border-teal/20 rounded-full motion-safe:animate-reticle-spin" />
      <div className="absolute inset-[5px] border-t border-teal/30 rounded-full" />
      <div className="absolute top-1/2 left-1/2 w-1 h-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-teal/50 motion-safe:animate-glow-pulse" />
    </div>
  )
}

export function AppShell() {
  const { sidebarOpen, toggleSidebar, docPanelOpen, toggleDocPanel, activeWorkspaceId } = useApp()

  const { data } = useQuery({
    queryKey: ['documents', activeWorkspaceId],
    queryFn: () => docsApi.list(activeWorkspaceId!).then((r) => r.data),
    enabled: !!activeWorkspaceId,
  })
  const docCount = data?.total ?? 0

  return (
    // `relative` is load-bearing: the HUD background layers are absolutely
    // positioned and must clip to this shell, not the document. `100dvh`
    // tracks the dynamic viewport so mobile browser chrome never crops the
    // composer.
    <div className="relative flex h-[100dvh] w-screen bg-void overflow-hidden">
      {/* ── Holographic-HUD environment ──────────────────────────── */}
      {/* Slowly scrolling alignment grid */}
      <div className="absolute inset-0 bg-grid motion-safe:animate-grid-scroll pointer-events-none opacity-70" />
      {/* Depth glows */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_70%_55%_at_50%_0%,rgba(31,240,230,0.06),transparent_70%)] pointer-events-none" />
      <div className="absolute -top-1/4 -left-1/4 w-[70vw] h-[70vh] rounded-full bg-[radial-gradient(circle,rgba(31,240,230,0.07),transparent_60%)] blur-3xl pointer-events-none motion-safe:animate-aurora" />
      <div className="absolute -bottom-1/3 -right-1/4 w-[60vw] h-[60vh] rounded-full bg-[radial-gradient(circle,rgba(157,107,255,0.06),transparent_60%)] blur-3xl pointer-events-none motion-safe:animate-aurora-slow" />
      {/* CRT scanlines + a single drifting scan sweep */}
      <div className="absolute inset-0 bg-scanlines pointer-events-none opacity-50 mix-blend-screen" />
      <div className="absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-teal/[0.07] to-transparent pointer-events-none motion-safe:animate-scanline" />
      {/* Edge vignette to focus the centre */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,transparent_52%,rgba(0,0,0,0.6))]" />
      {/* Corner reticles (hidden on small screens to reduce clutter) */}
      <CornerReticle className="-top-4 -right-4 hidden md:block" />
      <CornerReticle className="-bottom-4 -left-4 hidden md:block" />

      {/* Ctrl/Cmd+K command palette */}
      <CommandPalette />

      {/* Sidebar toggle (when collapsed) */}
      {!sidebarOpen && (
        <button
          onClick={toggleSidebar}
          aria-label="Open sidebar"
          className="absolute top-3.5 left-3.5 z-50 w-8 h-8 rounded-md hud-panel text-dim hover:text-teal flex items-center justify-center transition-colors"
        >
          <PanelLeft className="w-3.5 h-3.5" />
        </button>
      )}

      {/* Sidebar — overlay drawer on mobile, inline column on lg+ */}
      {sidebarOpen && (
        <>
          <div
            onClick={toggleSidebar}
            className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
            aria-hidden
          />
          <div className="fixed lg:relative z-40 lg:z-10 flex h-full">
            <Sidebar />
            <button
              onClick={toggleSidebar}
              aria-label="Collapse sidebar"
              className="absolute -right-3 top-3.5 z-10 w-6 h-6 rounded-full hud-panel text-dim hover:text-teal flex items-center justify-center transition-colors"
            >
              <PanelLeftClose className="w-3 h-3" />
            </button>
          </div>
        </>
      )}

      {/* Main content */}
      <main className="relative z-10 flex-1 flex overflow-hidden min-w-0">
        {/* Chat */}
        <div className="flex-1 overflow-hidden min-w-0">
          <ChatView />
        </div>

        {/* Documents panel — full-width drawer on mobile, inline rail on lg+ */}
        {docPanelOpen && (
          <>
            <div
              onClick={toggleDocPanel}
              className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
              aria-hidden
            />
            <div className="fixed inset-y-0 right-0 z-40 w-full max-w-sm lg:static lg:z-auto lg:w-80 shrink-0 border-l border-teal/10 hud-panel">
              <DocumentsPanel onClose={toggleDocPanel} />
            </div>
          </>
        )}

        {/* Documents toggle — anchored top-right so it never overlaps the
            composer at the bottom of the chat (issue #3). */}
        {!docPanelOpen && (
          <button
            onClick={toggleDocPanel}
            aria-label="Open documents"
            className="absolute top-3 right-3 z-40 flex items-center gap-2 h-9 px-3 sm:px-4 rounded-lg hud-panel hud-corners text-subtle hover:text-teal transition-all hover:shadow-teal"
          >
            <FileText className="w-3.5 h-3.5" />
            <span className="hidden sm:inline text-xs font-medium tracking-wide">Documents</span>
            {docCount > 0 && (
              <span className="ml-0.5 px-1.5 py-0.5 rounded-full bg-teal/10 border border-teal/25 text-teal font-mono text-2xs">
                {docCount}
              </span>
            )}
          </button>
        )}
      </main>
    </div>
  )
}
