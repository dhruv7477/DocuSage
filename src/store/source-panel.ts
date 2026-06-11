import { create } from 'zustand'
import type { CitedSource } from '@/api'

interface SourcePanelState {
  open: boolean
  source: CitedSource | null
  openSource: (source: CitedSource) => void
  close: () => void
}

export const useSourcePanel = create<SourcePanelState>((set) => ({
  open: false,
  source: null,
  openSource: (source) => set({ open: true, source }),
  close: () => set({ open: false, source: null }),
}))
