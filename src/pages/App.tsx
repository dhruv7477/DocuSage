import { useQuery } from '@tanstack/react-query'
import { PanelLeftClose, PanelLeft, FileText } from 'lucide-react'
import { useApp } from '@/store/app'
import { Sidebar } from '@/components/Sidebar'
import { ChatView } from '@/components/ChatView'
import { DocumentsPanel } from '@/components/DocumentsPanel'
import { documents as docsApi } from '@/api'

export function AppShell() {
  const { sidebarOpen, toggleSidebar, docPanelOpen, toggleDocPanel, activeWorkspaceId } = useApp()

  const { data } = useQuery({
    queryKey: ['documents', activeWorkspaceId],
    queryFn: () => docsApi.list(activeWorkspaceId!).then((r) => r.data),
    enabled: !!activeWorkspaceId,
  })
  const docCount = data?.total ?? 0

  return (
    <div className="flex h-screen w-screen bg-void overflow-hidden">
      {/* Grid background */}
      <div className="absolute inset-0 bg-grid bg-grid pointer-events-none opacity-100" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_20%_50%,rgba(0,212,200,0.03),transparent)] pointer-events-none" />

      {/* Sidebar toggle (always visible) */}
      {!sidebarOpen && (
        <button
          onClick={toggleSidebar}
          className="absolute top-3.5 left-3.5 z-50 w-7 h-7 rounded-md bg-surface border border-border text-dim hover:text-teal flex items-center justify-center transition-colors"
        >
          <PanelLeft className="w-3.5 h-3.5" />
        </button>
      )}

      {/* Sidebar */}
      {sidebarOpen && (
        <div className="relative z-10 flex">
          <Sidebar />
          <button
            onClick={toggleSidebar}
            className="absolute -right-3 top-3.5 z-10 w-6 h-6 rounded-full bg-surface border border-border text-dim hover:text-teal flex items-center justify-center transition-colors"
          >
            <PanelLeftClose className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* Main content */}
      <main className="relative flex-1 flex overflow-hidden">
        {/* Chat */}
        <div className="flex-1 overflow-hidden">
          <ChatView />
        </div>

        {/* Documents panel (slide-in) */}
        {docPanelOpen && (
          <div className="w-80 shrink-0 border-l border-border bg-surface/80 backdrop-blur-sm">
            <DocumentsPanel onClose={toggleDocPanel} />
          </div>
        )}

        {/* Floating Documents button */}
        {!docPanelOpen && (
          <button
            onClick={toggleDocPanel}
            className="absolute bottom-6 right-6 z-40 flex items-center gap-2 h-9 px-4 rounded-full bg-surface border border-border text-subtle hover:text-teal hover:border-teal/40 shadow-panel transition-all hover:shadow-teal/10"
          >
            <FileText className="w-3.5 h-3.5" />
            <span className="text-xs font-medium">Documents</span>
            {docCount > 0 && (
              <span className="ml-0.5 px-1.5 py-0.5 rounded-full bg-teal/10 border border-teal/20 text-teal font-mono text-2xs">
                {docCount}
              </span>
            )}
          </button>
        )}
      </main>
    </div>
  )
}
