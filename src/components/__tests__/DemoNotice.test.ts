import { describe, it, expect } from 'vitest'
import { demoNoticeLines } from '@/components/DemoNotice'
import type { DemoInfo } from '@/api'

const demo: DemoInfo = {
  enabled: true,
  max_documents: 5,
  max_upload_mb: 25,
  questions_per_day: 30,
  wipe_hour_utc: 3,
  llm_host: 'api.groq.com',
}

describe('demoNoticeLines', () => {
  it('states the limits, the nightly wipe and the third-party LLM', () => {
    const text = demoNoticeLines(demo).join(' ')
    expect(text).toContain('Do not upload anything private')
    expect(text).toContain('keep up to 5 documents, 25 MB of uploads in total, ask 30 questions a day')
    expect(text).toContain('deleted every day at 03:00 UTC')
    expect(text).toContain('sent to api.groq.com')
  })

  it('leaves out limits that are unlimited and an LLM that runs locally', () => {
    const text = demoNoticeLines({
      ...demo,
      max_documents: 0,
      max_upload_mb: 0,
      questions_per_day: 0,
      llm_host: null,
    }).join(' ')
    expect(text).not.toContain('You can')
    expect(text).not.toContain('sent to')
  })
})
