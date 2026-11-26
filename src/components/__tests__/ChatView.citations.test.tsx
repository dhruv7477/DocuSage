/**
 * A saved answer's citations: gpt-oss's 【n†p.2】 markers show as clickable
 * [n] chips, and "Check sources" lists exactly the cited sources under the
 * same numbers, so a chip and its source card always match. A source shows
 * its rank among the passages, not the reranker score as a "% match" (the
 * score is not calibrated). A source an older answer cited for facts its
 * text lacks is shown without a highlight, an answer that cites none of
 * the documents is labelled "Not from your documents", and one with a claim
 * whose citation was dropped carries a note.
 */

import { describe, it, expect, vi, afterAll, beforeAll, beforeEach } from 'vitest'
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
const pageImage = vi.hoisted(() =>
  vi.fn((..._args: unknown[]) => ({ data: undefined, isLoading: false, isError: false })),
)
vi.mock('@/hooks/usePageImage', () => ({ usePageImage: pageImage }))

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
      // Source 2 has the highest score but is not cited. Source 3 was cited
      // by an older version for a fact its text lacks: shown without a highlight.
      sources: [source(1, 0.5), source(2, 0.99), { ...source(3, 0.4), highlight: false }],
      created_at: '2026-11-23T10:00:05Z',
    },
    {
      id: 'm3',
      chat_id: 'chat-1',
      role: 'user',
      content: 'What does an I-9 auditor typically do?',
      token_count: null,
      created_at: '2026-11-23T10:01:00Z',
    },
    {
      id: 'm4',
      chat_id: 'chat-1',
      role: 'assistant',
      content: 'Auditors compare each form against its supporting documents.',
      token_count: null,
      sources: [],
      label: 'not_from_documents',
      created_at: '2026-11-23T10:01:05Z',
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

describe('source cards', () => {
  it('show the rank among the passages, not a percentage', async () => {
    await renderAnswer()
    fireEvent.click(screen.getByRole('button', { name: /Check sources/ }))

    expect(screen.getByText('rank 1 of 3')).toBeInTheDocument()
    expect(screen.getByText('rank 3 of 3')).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(/\d+% match/)
  })

  it('show the rank in the passage popover too', async () => {
    await renderAnswer()
    fireEvent.mouseEnter(screen.getByRole('button', { name: 'Source 1' }))

    const popover = await screen.findByRole('dialog')
    expect(popover).toHaveTextContent('rank 1 of 3')
    expect(popover.textContent).not.toMatch(/\d+%/)
  })

  it('show a source cited for facts its text lacks without a highlight', async () => {
    pageImage.mockClear()
    await renderAnswer()
    fireEvent.click(screen.getByRole('button', { name: /Check sources/ }))

    const calls = pageImage.mock.calls as unknown[][]
    // [1] is highlighted on its page; [3] is shown as the plain page.
    expect(calls).toContainEqual(['doc-1', 1, 'chunk-1', 2, true])
    expect(calls).toContainEqual(['doc-1', 3, undefined, 2, true])
    expect(calls).not.toContainEqual(['doc-1', 3, 'chunk-3', 2, true])
  })
})

describe('an answer that cites none of the documents', () => {
  it('is labelled "Not from your documents" and has no sources', async () => {
    await renderAnswer()
    const answer = screen.getByText(/Auditors compare each form/)
    const bubble = answer.closest('.max-w-2xl') as HTMLElement

    expect(bubble).toHaveTextContent('Not from your documents')
    expect(screen.getAllByText('Not from your documents')).toHaveLength(1)
    expect(screen.getAllByRole('button', { name: /Check sources/ })).toHaveLength(1)
  })
})

describe('an answer with a claim whose citation was dropped', () => {
  // Added for this block only, so the other tests keep one "Check sources".
  const partial: Message[] = [
    {
      id: 'm5',
      chat_id: 'chat-1',
      role: 'user',
      content: 'Which languages does the employee speak?',
      token_count: null,
      created_at: '2026-11-23T10:02:00Z',
    },
    {
      id: 'm6',
      chat_id: 'chat-1',
      role: 'assistant',
      content: 'The form is available in Spanish [1]. Employees may also speak Portuguese.',
      token_count: null,
      sources: [{ document_id: 'doc-1', chunk_id: 'chunk-9', source: 'i-9-instructions.pdf', page: 9, score: 0.4 }],
      label: 'parts_unlinked',
      created_at: '2026-11-23T10:02:05Z',
    },
  ]
  beforeAll(() => {
    MESSAGES.push(...partial)
  })
  afterAll(() => {
    MESSAGES.splice(-partial.length)
  })

  it('carries one note under the answer, and no other answer does', async () => {
    await renderAnswer()
    const answer = await screen.findByText(/The form is available in Spanish/)
    const bubble = answer.closest('.max-w-2xl') as HTMLElement

    expect(bubble).toHaveTextContent("Parts of this answer aren't linked to a source.")
    expect(screen.getAllByText("Parts of this answer aren't linked to a source.")).toHaveLength(1)
    expect(bubble).not.toHaveTextContent('Not from your documents')
  })
})
