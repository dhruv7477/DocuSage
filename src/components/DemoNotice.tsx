import { useQuery } from '@tanstack/react-query'
import { Info } from 'lucide-react'
import { demo, type DemoInfo } from '@/api'

/** Plain-language lines describing what a demo account can do and where data goes. */
export function demoNoticeLines(info: DemoInfo): string[] {
  const lines = [
    'This is a public demo. Do not upload anything private or confidential.',
    'New accounts get public-domain sample documents to try right away.',
  ]
  const limits: string[] = []
  if (info.max_documents) limits.push(`keep up to ${info.max_documents} documents`)
  if (info.max_upload_mb) limits.push(`${info.max_upload_mb} MB of uploads in total`)
  if (info.questions_per_day) limits.push(`ask ${info.questions_per_day} questions a day`)
  if (limits.length) lines.push(`You can ${limits.join(', ')}.`)
  if (info.site_questions_per_day) {
    lines.push(
      `The whole demo answers up to ${info.site_questions_per_day} questions a day across all visitors.`,
    )
  }
  if (info.wipe_hour_utc !== null) {
    const hour = String(info.wipe_hour_utc).padStart(2, '0')
    lines.push(`Uploaded documents are deleted every day at ${hour}:00 UTC.`)
  }
  if (info.llm_host) {
    lines.push(
      `Your questions and the matching document excerpts are sent to ${info.llm_host} to write the answers.`,
    )
  }
  return lines
}

/** Shown on the sign-up page when the server runs in demo mode. */
export function DemoNotice() {
  const { data } = useQuery({
    queryKey: ['demo-info'],
    queryFn: async () => (await demo.info()).data,
    staleTime: Infinity,
    retry: false,
  })
  if (!data?.enabled) return null
  return (
    <div
      role="note"
      className="mb-4 rounded-lg border border-amber/30 bg-amber/[0.06] p-4 text-xs leading-relaxed text-subtle"
    >
      <div className="mb-1.5 flex items-center gap-2 font-semibold text-bright">
        <Info className="h-3.5 w-3.5 text-amber" />
        Demo account
      </div>
      <ul className="list-disc space-y-1 pl-4">
        {demoNoticeLines(data).map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </div>
  )
}
