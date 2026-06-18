/**
 * ThinkingStatus — the Claude Code-style status line. Phrase choice is
 * deterministic from elapsed time, so we pin time with fake timers.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { ThinkingStatus } from '@/components/ThinkingStatus'

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('ThinkingStatus', () => {
  it('falls back to "Thinking…" when stage is null', () => {
    render(<ThinkingStatus stage={null} startedAt={Date.now()} />)
    expect(screen.getByText('Thinking…')).toBeInTheDocument()
  })

  it('shows a stage-appropriate phrase and a seconds counter', () => {
    render(<ThinkingStatus stage="retrieving" startedAt={Date.now()} />)
    // First phrase for the stage at t=0.
    expect(screen.getByText('Combing through your documents…')).toBeInTheDocument()
    expect(screen.getByText('· 0s')).toBeInTheDocument()
  })

  it('elapsed counter ticks once per second', () => {
    render(<ThinkingStatus stage="generating" startedAt={Date.now()} />)
    act(() => vi.advanceTimersByTime(3000))
    expect(screen.getByText('· 3s')).toBeInTheDocument()
  })

  it('rotates phrases as time passes', () => {
    render(<ThinkingStatus stage="validating" startedAt={Date.now()} />)
    expect(screen.getByText('Double-checking citations…')).toBeInTheDocument()
    act(() => vi.advanceTimersByTime(3000)) // past the 2.5s rotation window
    expect(screen.getByText('Verifying every claim…')).toBeInTheDocument()
  })

  it('clamps negative elapsed when startedAt is in the future', () => {
    render(<ThinkingStatus stage="analyzing" startedAt={Date.now() + 60_000} />)
    expect(screen.getByText('· 0s')).toBeInTheDocument()
  })
})
