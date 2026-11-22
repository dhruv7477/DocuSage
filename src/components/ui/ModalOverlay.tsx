import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'

/**
 * The dimmed backdrop behind a dialog. It centres the dialog when it fits
 * and scrolls when the dialog is taller than the window: the document
 * itself never scrolls (index.html hides its overflow for the app shell).
 *
 * It renders into document.body, so it covers the whole window wherever it
 * is opened from: inside the sidebar, whose backdrop blur would otherwise
 * make `position: fixed` relative to the sidebar, and whose stacking order
 * puts it under the chat beside it.
 *
 * Clicking anywhere outside the dialog calls `onClose`. That click lands on
 * a transparent button behind the dialog, so the dialog's own content needs
 * no click handling; give the dialog `relative` so it sits above it.
 */
export function ModalOverlay({ onClose, children }: { onClose: () => void; children: ReactNode }) {
  return createPortal(
    <div className="fixed inset-0 z-50 overflow-y-auto bg-void/70 backdrop-blur-sm">
      <div className="relative flex min-h-full items-center justify-center py-4">
        <button
          type="button"
          aria-label="Close"
          tabIndex={-1}
          className="absolute inset-0 cursor-default"
          onClick={onClose}
        />
        {children}
      </div>
    </div>,
    document.body,
  )
}
