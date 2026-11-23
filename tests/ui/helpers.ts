import type { Locator, Page, Route } from '@playwright/test'

export const DEMO_INFO = {
  enabled: true,
  max_documents: 5,
  max_upload_mb: 25,
  questions_per_day: 10,
  site_questions_per_day: 50,
  wipe_hour_utc: 3,
  llm_host: 'api.groq.com',
}

/**
 * Answer every API call: the demo info, any `responses` the test sets up
 * (keyed by path after /api/v1), and 404 for anything else.
 */
export async function mockApi(
  page: Page,
  { demo }: { demo: boolean },
  responses: Record<string, unknown> = {},
): Promise<void> {
  await page.route('**/api/v1/**', async (route: Route) => {
    const path = new URL(route.request().url()).pathname.replace(/^\/api\/v1/, '')
    if (path === '/demo') {
      await route.fulfill({ json: { ...DEMO_INFO, enabled: demo } })
    } else if (path in responses) {
      await route.fulfill({ json: responses[path] })
    } else {
      await route.fulfill({ status: 404, json: { error_code: 'NOT_FOUND', message: `not mocked: ${path}` } })
    }
  })
}

async function inViewport(page: Page, locator: Locator): Promise<boolean> {
  const box = await locator.boundingBox()
  const viewport = page.viewportSize()
  if (!box || !viewport) return false
  return box.y >= 0 && box.y + box.height <= viewport.height
}

/**
 * Whether a visitor can bring `locator` fully into view with the mouse
 * wheel (down by default; a negative `deltaY` scrolls up). Never
 * scrollIntoView: that also scrolls containers that hide their overflow,
 * which a visitor cannot.
 */
export async function reachableByWheel(
  page: Page,
  locator: Locator,
  {
    at,
    deltaY = 250,
    maxSteps = 15,
  }: { at?: { x: number; y: number }; deltaY?: number; maxSteps?: number } = {},
): Promise<boolean> {
  await locator.waitFor()
  const viewport = page.viewportSize()
  if (!viewport) throw new Error('no viewport')
  const point = at ?? { x: viewport.width / 2, y: viewport.height / 2 }
  await page.mouse.move(point.x, point.y)
  for (let step = 0; step < maxSteps; step++) {
    if (await inViewport(page, locator)) return true
    await page.mouse.wheel(0, deltaY)
    await page.waitForTimeout(100)
  }
  return inViewport(page, locator)
}

/** The scroll offset of the document and of every scrolled element. */
export async function scrollPosition(page: Page): Promise<string> {
  return page.evaluate(() =>
    [document.scrollingElement, ...Array.from(document.querySelectorAll('*'))]
      .filter((el): el is Element => el !== null && el.scrollTop > 0)
      .map((el) => `${el.tagName}.${el.className}:${el.scrollTop}`)
      .join(','),
  )
}

/** Wait until every finite animation (such as the fade-in of a form) has finished. */
export async function settle(page: Page): Promise<void> {
  await page.evaluate(() =>
    Promise.all(
      document
        .getAnimations()
        .filter((a) => a.effect?.getComputedTiming().iterations !== Infinity)
        // A cancelled animation (its element re-rendered) rejects; it is over too.
        .map((a) => a.finished.catch(() => undefined)),
    ),
  )
}

/**
 * Load the page with a root font size of `px`, as Chrome's font size setting
 * (Settings > Appearance > Font size) or Windows' text size setting does: every
 * rem-based size grows, while the zoom and the viewport stay the same.
 */
export async function setRootFontSize(page: Page, px: number): Promise<void> {
  await page.addInitScript((size) => {
    // Init scripts run before the <html> element exists.
    const apply = () => document.documentElement.style.setProperty('font-size', `${size}px`)
    if (document.documentElement) {
      apply()
    } else {
      new MutationObserver((_records, observer) => {
        if (!document.documentElement) return
        apply()
        observer.disconnect()
      }).observe(document, { childList: true })
    }
  }, px)
}

/** The root element's computed font size, in px. */
export async function rootFontSize(page: Page): Promise<number> {
  return page.evaluate(() => Number.parseFloat(getComputedStyle(document.documentElement).fontSize))
}
