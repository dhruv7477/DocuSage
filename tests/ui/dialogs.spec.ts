/**
 * Dialogs in the signed-in app scroll when they are taller than the window,
 * stay centred when they fit, and still close on a click outside them.
 *
 * Workspace settings grows with every member; before the fix a tall dialog
 * was cut off at both ends (centred in a backdrop that hid its overflow),
 * so neither its title nor its last members could be reached.
 *
 * The dialogs are opened from the sidebar, whose backdrop blur makes it the
 * box that `position: fixed` is relative to. Before the fix they were also
 * squeezed into the sidebar's width on desktop, under the chat beside it.
 */

import { test, expect, type Page } from '@playwright/test'
import { mockApi, reachableByWheel, settle } from './helpers'

const USER = {
  id: 'u0',
  email: 'person0@example.com',
  full_name: 'Person 0',
  role: 'user',
  is_active: true,
  is_email_verified: true,
}
const WORKSPACE = {
  id: 'ws1',
  name: 'Sample documents',
  description: null,
  owner_id: 'u0',
  created_at: '2026-11-01T00:00:00Z',
  updated_at: '2026-11-01T00:00:00Z',
}
// Enough members for the dialog to be taller than every tested viewport.
const MEMBERS = Array.from({ length: 16 }, (_, i) => ({
  id: `m${i}`,
  user_id: `u${i}`,
  workspace_id: 'ws1',
  role: i === 0 ? 'owner' : 'member',
  user_email: `person${i}@example.com`,
  user_full_name: `Person ${i}`,
  joined_at: '2026-11-01T00:00:00Z',
}))

async function openApp(page: Page): Promise<void> {
  await page.addInitScript((user) => {
    localStorage.setItem(
      'docusage-auth',
      JSON.stringify({ state: { user, accessToken: 'token', refreshToken: 'refresh' }, version: 0 }),
    )
    localStorage.setItem('access_token', 'token')
    localStorage.setItem(
      'docusage-app',
      JSON.stringify({
        state: { activeWorkspaceId: 'ws1', activeChatId: null, sidebarOpen: true, docPanelOpen: false },
        version: 0,
      }),
    )
  }, USER)
  await mockApi(page, { demo: true }, {
    '/auth/me': USER,
    '/workspaces': [{ ...WORKSPACE, member_count: MEMBERS.length }],
    '/workspaces/ws1': { ...WORKSPACE, member_count: MEMBERS.length, members: MEMBERS },
    '/documents': { documents: [], total: 0 },
    '/chat': [],
  })
  await page.goto('/')
}

async function openWorkspaceMembers(page: Page): Promise<void> {
  await page.locator('button', { hasText: 'Sample documents' }).first().click()
  // The settings icon only shows on hover.
  await page.getByTitle('Workspace settings').first().click({ force: true })
  await page.getByRole('button', { name: 'Members', exact: true }).click()
  await expect(page.getByText(`person${MEMBERS.length - 1}@example.com`)).toBeAttached()
  await settle(page)
}

test('a dialog taller than the window scrolls to both ends', async ({ page }) => {
  await openApp(page)
  await openWorkspaceMembers(page)

  const lastMember = page.getByText(`person${MEMBERS.length - 1}@example.com`)
  const title = page.getByRole('heading', { name: 'Workspace settings' })
  expect(await reachableByWheel(page, lastMember)).toBe(true)
  expect(await reachableByWheel(page, title, { deltaY: -250 })).toBe(true)
})

test('the invite button and the remove buttons fit inside the dialog', async ({ page }) => {
  await openApp(page)
  await openWorkspaceMembers(page)

  // On a phone the full email and the invite input used to set the
  // content's width; the dialog then cut off the invite and remove buttons.
  const panel = (await page.getByRole('heading', { name: 'Workspace settings' }).locator('xpath=../..').boundingBox())!
  const lastRow = page.getByText(`person${MEMBERS.length - 1}@example.com`).locator('xpath=../..')
  for (const button of [page.getByRole('button', { name: 'Invite' }), lastRow.getByRole('button')]) {
    const box = (await button.boundingBox())!
    expect(box.x + box.width).toBeLessThanOrEqual(panel.x + panel.width)
  }
})

test('a dialog that fits stays centred, and a click outside closes it', async ({ page }) => {
  await openApp(page)
  await page.getByRole('button', { name: 'Settings', exact: true }).click()
  const title = page.getByRole('heading', { name: 'Settings', exact: true })
  await expect(title).toBeVisible()
  await settle(page)

  // Centred in the window on both axes, not inside the sidebar that opened it.
  const viewport = page.viewportSize()!
  const dialog = (await title.locator('xpath=../..').boundingBox())!
  const below = viewport.height - (dialog.y + dialog.height)
  const right = viewport.width - (dialog.x + dialog.width)
  expect(Math.abs(dialog.y - below)).toBeLessThanOrEqual(1)
  expect(Math.abs(dialog.x - right)).toBeLessThanOrEqual(1)

  await page.mouse.click(4, viewport.height - 4)
  await expect(title).toBeHidden()
})
