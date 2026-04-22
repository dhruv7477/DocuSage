import { ButtonHTMLAttributes, forwardRef } from 'react'
import { clsx } from 'clsx'
import { Loader2 } from 'lucide-react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'ghost' | 'danger' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'ghost', size = 'md', loading, disabled, className, children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={clsx(
          'inline-flex items-center justify-center gap-2 font-body font-medium transition-all duration-150 select-none',
          'disabled:opacity-40 disabled:cursor-not-allowed',
          {
            // sizes
            'h-7 px-3 text-xs rounded': size === 'sm',
            'h-9 px-4 text-sm rounded-md': size === 'md',
            'h-11 px-6 text-base rounded-lg': size === 'lg',
            // variants
            'bg-teal text-void hover:bg-teal/90 shadow-teal': variant === 'primary',
            'bg-transparent text-ghost hover:text-base hover:bg-surface': variant === 'ghost',
            'bg-transparent border border-border text-base hover:border-teal/40 hover:text-teal': variant === 'outline',
            'bg-red/10 border border-red/20 text-red hover:bg-red/20': variant === 'danger',
          },
          className
        )}
        {...props}
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : children}
      </button>
    )
  }
)
Button.displayName = 'Button'
