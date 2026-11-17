/**
 * useSSE — the frontend half of the SSE event contract.
 *
 * The backend contract (src/api/v1/chat.py _sse_generator):
 *   thinking{stage} · route{intent} · token{content} · answer_final{content}
 *   · warning{code,message} · sources{sources} · done{message_id} · error{message}
 *
 * fetchEventSource is mocked; we drive the hook by invoking the captured
 * onmessage callback with synthetic events.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

const fetchEventSourceMock = vi.fn()

vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: (...args: unknown[]) => fetchEventSourceMock(...args),
}))

import { useSSE } from '@/hooks/useSSE'

type SSEOptions = {
  onmessage: (e: { data: string }) => void
  onerror: (err: Error) => void
  signal: AbortSignal
  headers: Record<string, string>
}

function startStream(onDone?: (id: string) => void) {
  const { result } = renderHook(() => useSSE())
  let captured: SSEOptions | null = null
  fetchEventSourceMock.mockImplementation((_url: string, opts: SSEOptions) => {
    captured = opts
    return new Promise(() => {}) // stays pending like a live stream
  })
  act(() => {
    result.current.start('/api/v1/chat/x/stream?query=q', onDone)
  })
  const send = (event: object) =>
    act(() => captured!.onmessage({ data: JSON.stringify(event) }))
  return { result, send, options: () => captured! }
}

// jsdom's localStorage can be flaky across versions — use a tiny stub.
const storage = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => void storage.set(k, v),
  removeItem: (k: string) => void storage.delete(k),
  clear: () => storage.clear(),
})

beforeEach(() => {
  fetchEventSourceMock.mockReset()
  storage.clear()
})

describe('useSSE event contract', () => {
  it('start() resets state and records startedAt', () => {
    const { result } = startStream()
    expect(result.current.streaming).toBe(true)
    expect(result.current.content).toBe('')
    expect(result.current.startedAt).not.toBeNull()
  })

  it('attaches the Authorization header when a token exists', () => {
    localStorage.setItem('access_token', 'jwt-123')
    const { options } = startStream()
    expect(options().headers).toEqual({ Authorization: 'Bearer jwt-123' })
  })

  it('thinking events drive the stage verbatim (server-owned)', () => {
    const { result, send } = startStream()
    for (const stage of ['analyzing', 'routing', 'retrieving', 'reranking', 'generating', 'validating']) {
      send({ type: 'thinking', stage })
      expect(result.current.stage).toBe(stage)
    }
  })

  it('route event sets intent', () => {
    const { result, send } = startStream()
    send({ type: 'route', intent: 'doc_summary' })
    expect(result.current.intent).toBe('doc_summary')
  })

  it('token events append content in order', () => {
    const { result, send } = startStream()
    send({ type: 'token', content: 'Hello ' })
    send({ type: 'token', content: 'world.' })
    expect(result.current.content).toBe('Hello world.')
  })

  it('answer_final replaces the streamed content wholesale', () => {
    const { result, send } = startStream()
    send({ type: 'token', content: 'Bad answer [9]' })
    send({ type: 'answer_final', content: 'Bad answer' })
    expect(result.current.content).toBe('Bad answer')
  })

  it('sources event stores the citation list', () => {
    const { result, send } = startStream()
    const sources = [{ document_id: 'd1', source: 'a.pdf', page: 3, score: 0.9 }]
    send({ type: 'sources', sources })
    expect(result.current.sources).toEqual(sources)
  })

  it('warning events accumulate', () => {
    const { result, send } = startStream()
    send({ type: 'warning', code: 'low_confidence', message: 'weak' })
    send({ type: 'warning', code: 'summary_truncated', message: 'cut' })
    expect(result.current.warnings).toEqual([
      { code: 'low_confidence', message: 'weak' },
      { code: 'summary_truncated', message: 'cut' },
    ])
  })

  it('done stops streaming, clears stage, and reports the message id', () => {
    const onDone = vi.fn()
    const { result, send } = startStream(onDone)
    send({ type: 'token', content: 'Hi' })
    send({ type: 'done', message_id: 'msg-42' })
    expect(result.current.streaming).toBe(false)
    expect(result.current.stage).toBeNull()
    expect(onDone).toHaveBeenCalledWith('msg-42')
  })

  it('error event surfaces the message and stops streaming', () => {
    const { result, send } = startStream()
    send({ type: 'error', message: 'Something went wrong. Please try again.' })
    expect(result.current.streaming).toBe(false)
    expect(result.current.error).toBe('Something went wrong. Please try again.')
  })

  it('malformed JSON events are ignored without crashing', () => {
    const { result, options } = startStream()
    act(() => options().onmessage({ data: 'not-json{{' }))
    expect(result.current.streaming).toBe(true)
    expect(result.current.error).toBeNull()
  })

  it('dismissError clears only the error', () => {
    const { result, send } = startStream()
    send({ type: 'token', content: 'partial' })
    send({ type: 'error', message: 'boom' })
    act(() => result.current.dismissError())
    expect(result.current.error).toBeNull()
    expect(result.current.content).toBe('partial')
  })
})

describe('useSSE refused streams', () => {
  type OpenOptions = SSEOptions & { onopen: (r: Response) => Promise<void> }

  async function openWith(response: Response) {
    const { result, options } = startStream()
    const opts = options() as OpenOptions
    let thrown: unknown = null
    await act(async () => {
      try {
        await opts.onopen(response)
      } catch (err) {
        thrown = err
        // fetchEventSource passes the onopen error to onerror next
        try {
          opts.onerror(err as Error)
        } catch {
          // onerror rethrows to stop retries
        }
      }
    })
    return { result, thrown }
  }

  it('shows the server message when the daily quota is used up', async () => {
    const { result, thrown } = await openWith(
      new Response(JSON.stringify({ error_code: 'QUOTA_EXCEEDED', message: 'Daily question limit reached (30 per day).' }), {
        status: 429,
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(thrown).not.toBeNull()
    expect(result.current.streaming).toBe(false)
    expect(result.current.error).toBe('Daily question limit reached (30 per day).')
  })

  it('falls back to a status-based message for a plain 429', async () => {
    const { result } = await openWith(new Response('slow down', { status: 429 }))
    expect(result.current.error).toBe('Too many requests. Please wait a moment and try again.')
  })

  it('accepts a real event stream', async () => {
    const { result, thrown } = await openWith(
      new Response('', { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    )
    expect(thrown).toBeNull()
    expect(result.current.streaming).toBe(true)
    expect(result.current.error).toBeNull()
  })
})
