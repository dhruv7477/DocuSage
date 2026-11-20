/**
 * ChatView's composer: Enter or the send button starts a stream for the
 * typed question and adds it to the thread at once; Shift+Enter and a
 * blank box send nothing.
 */

import { describe, it, expect, vi, beforeAll, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ChatView } from '@/components/ChatView'
import { useApp } from '@/store/app'

const sse = vi.hoisted(() => ({
  content: '',
  error: null,
  notice: null,
  sources: [],
  stage: null,
  startedAt: null,
  streaming: false,
  warnings: [],
  start: vi.fn(),
  reset: vi.fn(),
  dismissError: vi.fn(),
  dismissNotice: vi.fn(),
}))

vi.mock('@/hooks/useSSE', () => ({ useSSE: () => sse }))

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>()
  return {
    ...actual,
    chat: { ...actual.chat, messages: vi.fn().mockResolvedValue({ data: [] }) },
    documents: {
      ...actual.documents,
      list: vi.fn().mockResolvedValue({
        data: { documents: [{ id: 'doc-1', filename: 'policy.pdf', status: 'completed' }] },
      }),
    },
  }
})

const QUESTION = 'What is the leave policy?'
const STREAM_URL = '/api/v1/chat/chat-1/stream?query=What+is+the+leave+policy%3F'

beforeAll(() => {
  // jsdom has no element scrolling; ChatView scrolls its message list.
  Element.prototype.scrollTo = vi.fn()
})

beforeEach(() => {
  sse.start.mockClear()
  useApp.setState({ activeChatId: 'chat-1', activeWorkspaceId: 'ws-1' })
})

let client: QueryClient

async function renderComposer(): Promise<HTMLElement> {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <ChatView />
    </QueryClientProvider>,
  )
  // The empty state shows once the (empty) message list has loaded, so a
  // late fetch cannot replace the optimistic message.
  await screen.findByRole('heading', { name: 'Query your documents' })
  const box = screen.getByRole('textbox')
  await waitFor(() => expect(box).toBeEnabled()) // a ready document unlocks it
  return box
}

describe('ChatView composer', () => {
  it('sends the trimmed question on Enter and adds it to the thread', async () => {
    const box = await renderComposer()
    fireEvent.change(box, { target: { value: `  ${QUESTION}  ` } })
    fireEvent.keyDown(box, { key: 'Enter' })

    expect(sse.start).toHaveBeenCalledOnce()
    expect(sse.start).toHaveBeenCalledWith(STREAM_URL, expect.any(Function))
    expect(client.getQueryData(['messages', 'chat-1'])).toEqual([
      expect.objectContaining({ id: '__temp__', role: 'user', content: QUESTION }),
    ])
    expect(box).toHaveValue('')
  })

  it('sends with the send button', async () => {
    const box = await renderComposer()
    fireEvent.change(box, { target: { value: QUESTION } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    expect(sse.start).toHaveBeenCalledWith(STREAM_URL, expect.any(Function))
  })

  it('sends a starter question when it is clicked', async () => {
    await renderComposer()
    const starter = 'What are the main topics covered across all documents?'
    fireEvent.click(screen.getByText(starter))

    const [url] = sse.start.mock.calls[0]
    expect(url).toBe(`/api/v1/chat/chat-1/stream?${new URLSearchParams({ query: starter })}`)
  })

  it('does not send on Shift+Enter', async () => {
    const box = await renderComposer()
    fireEvent.change(box, { target: { value: QUESTION } })
    fireEvent.keyDown(box, { key: 'Enter', shiftKey: true })

    expect(sse.start).not.toHaveBeenCalled()
  })

  it('does not send a blank question', async () => {
    const box = await renderComposer()
    fireEvent.change(box, { target: { value: '   ' } })
    fireEvent.keyDown(box, { key: 'Enter' })

    expect(sse.start).not.toHaveBeenCalled()
  })
})
