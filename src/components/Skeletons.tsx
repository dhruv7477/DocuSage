/**
 * Shimmer skeleton loaders — shown while chats/messages load so the UI
 * never presents a blank pane or a lone spinner.
 */

function ShimmerBlock({ className }: { className?: string }) {
  return (
    <div
      className={`relative overflow-hidden rounded bg-surface ${className ?? ''}`}
    >
      <div className="absolute inset-0 motion-safe:animate-shimmer bg-gradient-to-r from-transparent via-white/[0.04] to-transparent" />
    </div>
  )
}

export function ChatListSkeleton() {
  return (
    <div className="space-y-1.5 px-1 py-1" aria-busy="true" aria-label="Loading chats">
      {[88, 64, 76, 52, 70].map((w, i) => (
        <div key={i} className="flex items-center gap-2 px-1 py-1">
          <ShimmerBlock className="w-3.5 h-3.5 shrink-0" />
          <ShimmerBlock className="h-3" />
          <div style={{ width: `${w}%` }}>
            <ShimmerBlock className="h-3 w-full" />
          </div>
        </div>
      ))}
    </div>
  )
}

export function MessagesSkeleton() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-6" aria-busy="true" aria-label="Loading messages">
      {[
        { user: true, w: 'w-2/5' },
        { user: false, w: 'w-4/5' },
        { user: true, w: 'w-1/3' },
        { user: false, w: 'w-3/5' },
      ].map((row, i) => (
        <div key={i} className={`flex gap-3 ${row.user ? 'flex-row-reverse' : ''}`}>
          <ShimmerBlock className="w-7 h-7 rounded-lg shrink-0" />
          <div className={`${row.w} space-y-2`}>
            <ShimmerBlock className="h-4 w-full rounded-xl" />
            <ShimmerBlock className="h-4 w-3/4 rounded-xl" />
            {!row.user && <ShimmerBlock className="h-4 w-1/2 rounded-xl" />}
          </div>
        </div>
      ))}
    </div>
  )
}
