/**
 * Citation helpers: gpt-oss's full-width markers (【3†p.2】) become [n],
 * and an answer splits into text and the numbers of the sources it cites.
 */

import { describe, it, expect } from 'vitest'
import { citedSourceNumbers, normalizeCitations, splitCitations } from '@/components/citations'

describe('normalizeCitations', () => {
  it('rewrites full-width markers as [n]', () => {
    expect(normalizeCitations('three business days【1†p.2】【3†p.2】【8†p.4】.'))
      .toBe('three business days[1][3][8].')
  })

  it('handles bare numbers and lists', () => {
    expect(normalizeCitations('A【2】 and B【1, 3】.')).toBe('A[2] and B[1][3].')
  })

  it('drops a dagger marker without a number', () => {
    expect(normalizeCitations('Fact【source†L2-L4】.')).toBe('Fact.')
  })

  it('leaves ordinary full-width brackets alone', () => {
    const text = 'Brackets [like this], 【注意】, 【2023年】 and an unclosed 【1 stay'
    expect(normalizeCitations(text)).toBe(text)
  })
})

describe('splitCitations', () => {
  it('splits text and in-range markers, dropping out-of-range ones', () => {
    expect(splitCitations('Due in 3 days [1][3]. See [9].', 3)).toEqual([
      { text: 'Due in 3 days ' },
      { source: 1 },
      { source: 3 },
      { text: '. See ' },
      { text: '.' },
    ])
  })

  it('normalizes full-width markers first', () => {
    expect(splitCitations('Due【2†p.1】.', 2)).toEqual([{ text: 'Due' }, { source: 2 }, { text: '.' }])
  })
})

describe('citedSourceNumbers', () => {
  it('lists each in-range source once, in order of first citation', () => {
    expect(citedSourceNumbers('A [3] b [1] c [3] d [7]【1†p.2】', 3)).toEqual([3, 1])
  })

  it('is empty when nothing is cited', () => {
    expect(citedSourceNumbers('No markers here.', 4)).toEqual([])
  })
})
