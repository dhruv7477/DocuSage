import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AppState {
  activeWorkspaceId: string | null
  activeChatId: string | null
  sidebarOpen: boolean
  docPanelOpen: boolean
  setWorkspace: (id: string | null) => void
  setChat: (id: string | null) => void
  toggleSidebar: () => void
  toggleDocPanel: () => void
}

export const useApp = create<AppState>()(
  persist(
    (set) => ({
      activeWorkspaceId: null,
      activeChatId: null,
      sidebarOpen: true,
      docPanelOpen: false,

      setWorkspace: (id) => set({ activeWorkspaceId: id, activeChatId: null }),
      setChat: (id) => set({ activeChatId: id }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      toggleDocPanel: () => set((s) => ({ docPanelOpen: !s.docPanelOpen })),
    }),
    {
      name: 'docusage-app',
      partialize: (s) => ({ activeWorkspaceId: s.activeWorkspaceId }),
    }
  )
)
