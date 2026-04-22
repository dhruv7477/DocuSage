import { InputHTMLAttributes, forwardRef } from 'react'
import { clsx } from 'clsx'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  hint?: string
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, hint, className, ...props }, ref) => {
    return (
      <div className="flex flex-col gap-1.5">
        {label && (
          <label className="text-xs font-mono text-subtle uppercase tracking-widest">
            {label}
          </label>
        )}
        <input
          ref={ref}
          className={clsx(
            'h-10 w-full rounded-md bg-surface border px-3 text-sm text-base font-body',
            'placeholder:text-dim',
            'outline-none transition-all duration-150',
            error
              ? 'border-red/50 focus:border-red focus:shadow-[0_0_0_1px_rgba(255,77,106,0.3)]'
              : 'border-border focus:border-teal/60 focus:shadow-[0_0_0_1px_rgba(0,212,200,0.2)]',
            className
          )}
          {...props}
        />
        {(error || hint) && (
          <p className={clsx('text-xs', error ? 'text-red' : 'text-dim')}>
            {error ?? hint}
          </p>
        )}
      </div>
    )
  }
)
Input.displayName = 'Input'
