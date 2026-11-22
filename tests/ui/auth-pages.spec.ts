/**
 * The sign-in and sign-up pages scroll whenever their content is taller
 * than the viewport, and stay centred, without scrolling, when it fits.
 *
 * The document never scrolls (index.html hides its overflow for the app
 * shell). The sign-up form with the demo notice is taller than a 1366x768
 * window, and taller than 1920x936 with a larger default font size, so its
 * button and the link below it could not be reached.
 */

import { test, expect, type Page } from '@playwright/test'
import { mockApi, reachableByWheel, rootFontSize, scrollPosition, settle, setRootFontSize } from './helpers'

// Chrome's "Large" font size: 20 px instead of 16.
const LARGE_FONT_PX = 20

const PAGES = [
  { name: 'sign-up', path: '/register', heading: 'Create account', submit: 'Create Account', lastLink: 'Sign in' },
  { name: 'sign-in', path: '/login', heading: 'Welcome back', submit: 'Sign In', lastLink: 'Create one' },
]

async function open(page: Page, path: string, heading: string): Promise<void> {
  await mockApi(page, { demo: true })
  await page.goto(path)
  await expect(page.getByRole('heading', { name: heading })).toBeVisible()
  if (path === '/register') await expect(page.getByRole('note')).toContainText('Demo account')
  await settle(page)
}

for (const auth of PAGES) {
  test.describe(`${auth.name} page`, () => {
    test('the button and the link below it can be scrolled to', async ({ page }) => {
      await open(page, auth.path, auth.heading)

      expect(await reachableByWheel(page, page.getByRole('button', { name: auth.submit, exact: true }))).toBe(true)
      expect(await reachableByWheel(page, page.getByRole('link', { name: auth.lastLink, exact: true }))).toBe(true)
    })

    test('with a large default font size, everything can still be scrolled to', async ({ page }) => {
      await setRootFontSize(page, LARGE_FONT_PX)
      await open(page, auth.path, auth.heading)
      expect(await rootFontSize(page)).toBe(LARGE_FONT_PX)

      expect(await reachableByWheel(page, page.getByRole('button', { name: auth.submit, exact: true }))).toBe(true)
      expect(await reachableByWheel(page, page.getByRole('link', { name: auth.lastLink, exact: true }))).toBe(true)
    })

    test('the logo never overlaps the form', async ({ page }) => {
      await setRootFontSize(page, LARGE_FONT_PX)
      await open(page, auth.path, auth.heading)
      expect(await rootFontSize(page)).toBe(LARGE_FONT_PX)

      const logo = await page.getByText('DocuSage', { exact: true }).first().boundingBox()
      const form = await page.locator('.animate-fade-up').boundingBox()
      expect(logo && form).toBeTruthy()
      expect(logo!.y + logo!.height).toBeLessThanOrEqual(form!.y)
    })
  })
}

test('a sign-in form that fits stays centred and the page does not scroll', async ({ page }) => {
  await open(page, '/login', 'Welcome back')
  const viewport = page.viewportSize()!
  const form = (await page.locator('.animate-fade-up').boundingBox())!

  // Centred: the space above the form equals the space below it.
  const below = viewport.height - (form.y + form.height)
  expect(Math.abs(form.y - below)).toBeLessThanOrEqual(1)

  // Nothing to scroll: the wheel moves nothing.
  const before = await scrollPosition(page)
  await page.mouse.move(viewport.width / 2, viewport.height / 2)
  await page.mouse.wheel(0, 600)
  await page.waitForTimeout(200)
  expect(await scrollPosition(page)).toBe(before)
})
