import { useCallback, useRef, useState } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import type { CitedSource } from '@/api'

export type WarningCode = 'low_confidence' | 'no_documents' | 'partial_context' | string

export interface StreamWarning {
  code: WarningCode
  message: string
}

interface StreamState {
  streaming: boolean
  stage: 'retrieving' | 'generating' | null
  content: string
  sources: CitedSource[]
  warnings: StreamWarning[]
  error: string | null
}

interface UseSSEReturn extends StreamState {
  start: (url: string, onDone?: (messageId: string) => void) => void
  reset: () => void
  dismissError: () => void
}

const initialState: StreamState = {
  streaming: false,
  stage: null,
  content: '',
  sources: [],
  warnings: [],
  error: null,
}

export function useSSE(): UseSSEReturn {
  const [state, setState] = useState<StreamState>(initialState)
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setState(initialState)
  }, [])

  const dismissError = useCallback(() => {
    setState((s) => ({ ...s, error: null }))
  }, [])

  const start = useCallback((url: string, onDone?: (id: string) => void) => {
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl

    setState({ ...initialState, streaming: true })

    const token = localStorage.getItem('access_token')

    fetchEventSource(url, {
      signal: ctrl.signal,
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      onmessage(e) {
        try {
          const event = JSON.parse(e.data)
          switch (event.type) {
            case 'thinking':
              setState((s) => ({ ...s, stage: event.stage ?? null }))
              break
            case 'token':
              setState((s) => ({ ...s, stage: 'generating', content: s.content + event.content }))
              break
            case 'sources':
              setState((s) => ({ ...s, sources: event.sources }))
              break
            case 'warning':
              setState((s) => ({
                ...s,
                warnings: [
                  ...s.warnings,
                  { code: event.code ?? 'unknown', message: event.message ?? '' },
                ],
              }))
              break
            case 'done':
              setState((s) => ({ ...s, streaming: false, stage: null }))
              ctrl.abort()
              onDone?.(event.message_id)
              break
            case 'error':
              setState((s) => ({
                ...s,
                streaming: false,
                stage: null,
                error: event.message ?? 'The assistant could not respond.',
              }))
              ctrl.abort()
              break
          }
        } catch {
          // ignore parse errors
        }
      },
      onerror(err) {
        if ((err as Error).name !== 'AbortError') {
          setState((s) => ({
            ...s,
            streaming: false,
            error: 'Stream connection lost. Check that the backend is running and try again.',
          }))
        }
        throw err  // rethrow to stop fetchEventSource retrying
      },
    }).catch(() => {
      // fetchEventSource rejects on abort or onerror — safe to ignore
    })
  }, [])

  return { ...state, start, reset, dismissError }
}
