import { useCallback, useRef, useState } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import type { CitedSource } from '@/api'

export type WarningCode =
  | 'low_confidence'
  | 'no_documents'
  | 'model_abstained'
  | 'abstention_mismatch'
  | 'summary_truncated'
  | string

export interface StreamWarning {
  code: WarningCode
  message: string
}

export type ThinkingStage =
  | 'analyzing'
  | 'routing'
  | 'retrieving'
  | 'reranking'
  | 'generating'
  | 'summarizing'
  | 'validating'
  | null

interface StreamState {
  streaming: boolean
  stage: ThinkingStage
  intent: string | null
  startedAt: number | null
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

/** The server refused the stream before it started (quota, rate limit, access). */
class StreamRefusedError extends Error {}

/** User-facing message for a non-stream response, preferring the server's own. */
export async function refusalMessage(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (body && typeof body.message === 'string' && body.message) return body.message
  } catch {
    // not JSON: fall through to a message based on the status
  }
  if (response.status === 429) return 'Too many requests. Please wait a moment and try again.'
  if (response.status === 401) return 'Your session has expired. Please sign in again.'
  return `The assistant could not respond (HTTP ${response.status}).`
}

const initialState: StreamState = {
  streaming: false,
  stage: null,
  intent: null,
  startedAt: null,
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

    setState({ ...initialState, streaming: true, startedAt: Date.now() })

    const token = localStorage.getItem('access_token')

    fetchEventSource(url, {
      signal: ctrl.signal,
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      // Default is false, which ABORTS the stream when the tab is hidden
      // and silently retries on return — every retry re-sends the GET,
      // which re-saves the user message and restarts the whole pipeline.
      openWhenHidden: true,
      // A closed connection must surface as an error, never auto-retry:
      // retrying this GET has side effects (duplicate user messages).
      // Throwing stops the library's built-in retry loop.
      onclose() {
        setState((s) => {
          if (!s.streaming) return s  // closed after done/error — normal
          return {
            ...s,
            streaming: false,
            stage: null,
            error: s.error ?? 'The connection closed before the answer finished. Please try again.',
          }
        })
        throw new Error('SSE connection closed by server')
      },
      // Errors raised before streaming starts (quota, rate limit, access)
      // arrive as a JSON response. Show the server's message.
      async onopen(response) {
        const type = response.headers.get('content-type') ?? ''
        if (response.ok && type.startsWith('text/event-stream')) return
        const message = await refusalMessage(response)
        setState((s) => ({ ...s, streaming: false, stage: null, error: message }))
        throw new StreamRefusedError(message)
      },
      onmessage(e) {
        try {
          const event = JSON.parse(e.data)
          switch (event.type) {
            case 'thinking':
              // The server owns stage transitions — trust them verbatim.
              setState((s) => ({ ...s, stage: event.stage ?? null }))
              break
            case 'route':
              setState((s) => ({ ...s, intent: event.intent ?? null }))
              break
            case 'token':
              setState((s) => ({ ...s, content: s.content + event.content }))
              break
            case 'answer_final':
              // Validator corrected the answer — replace wholesale.
              setState((s) => ({ ...s, content: event.content ?? s.content }))
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
        if ((err as Error).name !== 'AbortError' && !(err instanceof StreamRefusedError)) {
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
