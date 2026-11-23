import { defineConfig } from '@playwright/test'

// Layout checks in a real browser: jsdom (Vitest) does no layout, so it
// cannot tell whether a page scrolls. The tests run against the production
// build served by `vite preview` (run `npm run build` first) and answer
// every API call themselves, so no backend is needed.
export default defineConfig({
  testDir: 'tests/ui',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    browserName: 'chromium',
    // The background animations are motion-safe; this keeps screenshots stable.
    reducedMotion: 'reduce',
    // Kept in test-results/; CI uploads them when a test fails.
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop-1920x936', use: { viewport: { width: 1920, height: 936 } } },
    { name: 'laptop-1366x768', use: { viewport: { width: 1366, height: 768 } } },
    {
      name: 'phone-390x844',
      use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 3 },
    },
  ],
  webServer: {
    command: 'npm run preview -- --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
  },
})
