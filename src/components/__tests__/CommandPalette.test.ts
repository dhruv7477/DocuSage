/**
 * fuzzyScore — the command palette's match ranking. Lower = better;
 * substrings beat subsequences; null = no match.
 */

import { describe, it, expect } from 'vitest'
import { fuzzyScore } from '@/components/CommandPalette'

describe('fuzzyScore', () => {
  it('empty needle matches everything with the best score', () => {
    expect(fuzzyScore('', 'anything')).toBe(0)
  })

  it('substring match scores by position (earlier is better)', () => {
    expect(fuzzyScore('chat', 'chat with docs')).toBe(0)
    expect(fuzzyScore('docs', 'chat with docs')).toBe(10)
  })

  it('is case-insensitive', () => {
    expect(fuzzyScore('REPORT', 'Quarterly Report.docx')).not.toBeNull()
  })

  it('subsequence match ranks after any substring match', () => {
    const substring = fuzzyScore('new', 'new chat')!
    const subsequence = fuzzyScore('nct', 'new chat')!
    expect(substring).toBeLessThan(subsequence)
    expect(subsequence).toBeGreaterThanOrEqual(100)
  })

  it('tighter subsequences score better than scattered ones', () => {
    const tight = fuzzyScore('abc', 'xabcx')!
    const scattered = fuzzyScore('abc', 'a__b__c')!
    expect(tight).toBeLessThan(scattered)
  })

  it('returns null when characters are missing', () => {
    expect(fuzzyScore('xyz', 'workspace')).toBeNull()
  })
})
