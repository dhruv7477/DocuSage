/**
 * ModalOverlay: a click outside the dialog closes it; a click inside does not;
 * it renders into document.body.
 * (Scrolling needs real layout; tests/ui/dialogs.spec.ts checks it in Chromium.)
 */

import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ModalOverlay } from '@/components/ui/ModalOverlay'

function renderDialog(onClose: () => void) {
  render(
    <ModalOverlay onClose={onClose}>
      <div className="relative">
        <p>Dialog content</p>
        <button type="button">Save</button>
      </div>
    </ModalOverlay>,
  )
}

describe('ModalOverlay', () => {
  it('closes on a click outside the dialog', () => {
    const onClose = vi.fn()
    renderDialog(onClose)
    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('does not close on a click inside the dialog', () => {
    const onClose = vi.fn()
    renderDialog(onClose)
    fireEvent.click(screen.getByText('Dialog content'))
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(onClose).not.toHaveBeenCalled()
  })

  it('keeps the backdrop button out of the tab order', () => {
    renderDialog(vi.fn())
    expect(screen.getByRole('button', { name: 'Close' })).toHaveAttribute('tabindex', '-1')
  })

  it('renders into document.body, outside the component that opens it', () => {
    const { container } = render(
      <aside>
        <ModalOverlay onClose={vi.fn()}>
          <p>Dialog content</p>
        </ModalOverlay>
      </aside>,
    )
    const content = screen.getByText('Dialog content')
    expect(container.contains(content)).toBe(false)
    expect(document.body.contains(content)).toBe(true)
  })
})
