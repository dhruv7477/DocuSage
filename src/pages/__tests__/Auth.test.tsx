/**
 * The sign-in and sign-up pages render their "// ..." header labels as
 * text, not as stray JSX comments.
 */

import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { LoginPage, RegisterPage } from '@/pages/Auth'

vi.mock('@/api', () => ({
  auth: { login: vi.fn(), register: vi.fn(), me: vi.fn(), googleLogin: vi.fn() },
  demo: { info: vi.fn().mockResolvedValue({ data: { enabled: false } }) },
}))

function renderPage(page: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>{page}</MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('auth page header labels', () => {
  it('shows the sign-in label', () => {
    renderPage(<LoginPage />)
    expect(screen.getByText('// AUTHENTICATE')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Welcome back' })).toBeInTheDocument()
  })

  it('shows the sign-up label', () => {
    renderPage(<RegisterPage />)
    expect(screen.getByText('// NEW_INSTANCE')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Create account' })).toBeInTheDocument()
  })
})
