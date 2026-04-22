import { useCallback, useRef, useState } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import type { CitedSource } from '@/api'

interface StreamState {
  streaming: boolean
  content: string
  sources: CitedSource[]
  error: string | null
}

interface UseSSEReturn extends StreamState {
  start: (url: string, onDone?: (messageId: string) => void) => void
  reset: () => void
}

export function useSSE(): UseSSEReturn {
  const [state, setState] = useState<StreamState>({
    streaming: false,
    content: '',
    sources: [],
    error: null,
  })
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setState({ streaming: false, content: '', sources: [], error: null })
  }, [])

  const start = useCallback((url: string, onDone?: (id: string) => void) => {
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl

    setState({ streaming: true, content: '', sources: [], error: null })

    const token = localStorage.getItem('access_token')

    fetchEventSource(url, {
      signal: ctrl.signal,
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      onmessage(e) {
        try {
          const event = JSON.parse(e.data)
          switch (event.type) {
            case 'token':
              setState((s) => ({ ...s, content: s.content + event.content }))
              break
            case 'sources':
              setState((s) => ({ ...s, sources: event.sources }))
              break
            case 'done':
              setState((s) => ({ ...s, streaming: false }))
              ctrl.abort()
              onDone?.(event.message_id)
              break
            case 'error':
              setState((s) => ({ ...s, streaming: false, error: event.message }))
              ctrl.abort()
              break
          }
        } catch {
          // ignore parse errors
        }
      },
      onerror(err) {
        if ((err as Error).name !== 'AbortError') {
          setState((s) => ({ ...s, streaming: false, error: 'Stream connection lost' }))
        }
        throw err  // rethrow to stop fetchEventSource retrying
      },
    }).catch(() => {
      // fetchEventSource rejects on abort or onerror — safe to ignore
    })
  }, [])

  return { ...state, start, reset }
}
