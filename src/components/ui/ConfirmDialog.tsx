import { Button } from './Button'
import { ModalOverlay } from './ModalOverlay'

interface ConfirmDialogProps {
  title: string
  description: string
  confirmLabel?: string
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
}

export function ConfirmDialog({
  title,
  description,
  confirmLabel = 'Confirm',
  danger = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <ModalOverlay onClose={onCancel}>
      <div className="relative bg-panel border border-border rounded-xl shadow-panel w-full max-w-sm mx-4 p-6">
        <h2 className="font-display text-base font-semibold text-bright mb-2">{title}</h2>
        <p className="text-sm text-subtle mb-6 leading-relaxed">{description}</p>
        <div className="flex gap-2 justify-end">
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            variant={danger ? 'danger' : 'primary'}
            size="sm"
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </ModalOverlay>
  )
}
