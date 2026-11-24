/**
 * A saved answer's citations: gpt-oss's 【n†p.2】 markers show as clickable
 * [n] chips, and "Check sources" lists exactly the cited sources under the
 * same numbers, so a chip and its source card always match.
 */

import { describe, it, expect, vi, beforeAll, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ChatView } from '@/components/ChatView'
import { useApp } from '@/store/app'
import type { CitedSource, Message } from '@/api'

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
vi.mock('@/hooks/usePageImage', () => ({
  usePageImage: () => ({ data: undefined, isLoading: false, isError: false }),
}))

// vi.mock factories run before the file's other code, so the fixtures
// they return are built in vi.hoisted.
const { MESSAGES } = vi.hoisted(() => {
  function source(n: number, score: number): CitedSource {
    return {
      document_id: 'doc-1',
      chunk_id: `chunk-${n}`,
      source: 'i-9-instructions.pdf',
      page: n,
      score,
      heading_path: [],
      text: `Passage ${n}`,
    }
  }

  const messages: Message[] = [
    {
      id: 'm1',
      chat_id: 'chat-1',
      role: 'user',
      content: 'When must an employer complete Section 2?',
      token_count: null,
      created_at: '2026-11-23T10:00:00Z',
    },
    {
      id: 'm2',
      chat_id: 'chat-1',
      role: 'assistant',
      content: 'Within three business days【1†p.2】【3†p.2】.',
      token_count: null,
      // Source 2 has the highest score but is not cited.
      sources: [source(1, 0.5), source(2, 0.99), source(3, 0.4)],
      created_at: '2026-11-23T10:00:05Z',
    },
  ]
  return { MESSAGES: messages }
})

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>()
  return {
    ...actual,
    chat: { ...actual.chat, messages: vi.fn().mockResolvedValue({ data: MESSAGES }) },
    documents: {
      ...actual.documents,
      list: vi.fn().mockResolvedValue({
        data: { documents: [{ id: 'doc-1', filename: 'i-9-instructions.pdf', status: 'completed' }] },
      }),
    },
  }
})

beforeAll(() => {
  Element.prototype.scrollTo = vi.fn()
})

beforeEach(() => {
  useApp.setState({ activeChatId: 'chat-1', activeWorkspaceId: 'ws-1' })
})

async function renderAnswer(): Promise<void> {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <ChatView />
    </QueryClientProvider>,
  )
  await screen.findByText(/Within three business days/)
}

describe('citations in a saved answer', () => {
  it('shows [n] chips instead of full-width markers', async () => {
    await renderAnswer()

    expect(screen.getByRole('button', { name: 'Source 1' })).toHaveTextContent('[1]')
    expect(screen.getByRole('button', { name: 'Source 3' })).toHaveTextContent('[3]')
    expect(screen.queryByRole('button', { name: 'Source 2' })).toBeNull()
    expect(document.body.textContent).not.toContain('【')
  })

  it('lists the cited sources under the same numbers', async () => {
    await renderAnswer()

    const toggle = screen.getByRole('button', { name: /Check sources/ })
    expect(toggle).toHaveTextContent('2')
    fireEvent.click(toggle)

    // Each number appears twice: its chip and its source card.
    expect(screen.getAllByText('[1]')).toHaveLength(2)
    expect(screen.getAllByText('[3]')).toHaveLength(2)
    expect(screen.queryByText('[2]')).toBeNull()
  })
})
