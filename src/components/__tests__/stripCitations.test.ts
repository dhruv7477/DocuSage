/**
 * stripCitations — removes [n] markers from an answer and tidies the
 * spacing and punctuation they leave behind.
 */

import { describe, it, expect } from 'vitest'
import { stripCitations } from '@/components/ChatView'

describe('stripCitations', () => {
  it('removes citation markers and the space before the punctuation', () => {
    expect(stripCitations('Leave is 20 days [1]. It is paid [2] [3].'))
      .toBe('Leave is 20 days. It is paid.')
  })

  it('hides a half-streamed marker at the end', () => {
    expect(stripCitations('Leave is 20 days [1')).toBe('Leave is 20 days ')
    expect(stripCitations('Leave is 20 days [')).toBe('Leave is 20 days ')
  })

  it('removes full-width markers, complete or half-streamed', () => {
    expect(stripCitations('Leave is 20 days【1†p.2】【3†p.2】.')).toBe('Leave is 20 days.')
    expect(stripCitations('Leave is 20 days【1†p')).toBe('Leave is 20 days')
  })

  it('collapses runs of spaces and tabs', () => {
    expect(stripCitations('one  \t two')).toBe('one two')
  })

  it('removes newlines before punctuation but keeps them elsewhere', () => {
    expect(stripCitations('First line\n\n, second')).toBe('First line, second')
    expect(stripCitations('First line\n\nSecond line')).toBe('First line\n\nSecond line')
  })

  it('removes whitespace before punctuation at the very start', () => {
    expect(stripCitations(' . start')).toBe('. start')
  })

  it('leaves text without markers unchanged', () => {
    expect(stripCitations('No citations here, at all!')).toBe('No citations here, at all!')
  })

  it('handles a long whitespace run that no punctuation follows', () => {
    const text = `a${'\n'.repeat(50_000)}b`
    expect(stripCitations(text)).toBe(text)
  })
})
