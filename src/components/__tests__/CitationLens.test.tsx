/**
 * highlightQueryTerms — the lens's query-term marker. Pure function,
 * tested by rendering its output and inspecting the <mark> elements.
 */

import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import { highlightQueryTerms } from '@/components/CitationLens'

function marksFor(text: string, query: string): string[] {
  const { container } = render(<div>{highlightQueryTerms(text, query)}</div>)
  return Array.from(container.querySelectorAll('mark')).map((m) => m.textContent ?? '')
}

describe('highlightQueryTerms', () => {
  it('marks query terms case-insensitively', () => {
    expect(marksFor('The Employee must present documents.', 'employee documents'))
      .toEqual(['Employee', 'documents'])
  })

  it('returns plain text when the query is empty', () => {
    expect(marksFor('Nothing to mark here.', '   ')).toEqual([])
  })

  it('ignores stop words and short terms', () => {
    // "the", "and" are stop words; "of" is too short — only "deadline" marks.
    expect(marksFor('The deadline and the rest of it.', 'the and of deadline'))
      .toEqual(['deadline'])
  })

  it('escapes regex special characters in terms', () => {
    expect(marksFor('Supports c++ syntax.', 'c++')).toEqual(['c++'])
  })

  it('marks repeated occurrences', () => {
    expect(marksFor('policy here, policy there', 'policy'))
      .toEqual(['policy', 'policy'])
  })

  it('preserves the full text around the marks', () => {
    const { container } = render(
      <div>{highlightQueryTerms('alpha beta gamma', 'beta')}</div>
    )
    expect(container.textContent).toBe('alpha beta gamma')
  })

  it('deduplicates repeated query terms', () => {
    expect(marksFor('refund window', 'refund refund refund'))
      .toEqual(['refund'])
  })
})
