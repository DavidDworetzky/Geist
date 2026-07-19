import { defineConfig } from '@playwright/test';

const backendPort = Number(process.env.GEIST_E2E_BACKEND_PORT ?? 5100);
const frontendPort = 3100;

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  reporter: process.env.CI
    ? [['line'], ['html', { open: 'never' }]]
    : 'line',
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    channel: process.env.GEIST_E2E_USE_BUNDLED_CHROMIUM ? undefined : 'chrome',
    headless: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: [
    ...(process.env.GEIST_E2E_EXTERNAL_BACKEND
      ? []
      : [
          {
            command: 'uv run python -m tests.e2e_server',
            cwd: '../..',
            url: `http://127.0.0.1:${backendPort}/docs`,
            reuseExistingServer: false,
            timeout: 120_000,
            env: {
              ...process.env,
              PYTHONPATH: '.',
              GEIST_DATABASE_PROVIDER: 'sqlite',
              SQLITE_DATABASE_PATH: `/tmp/geist-playwright-${process.pid}.sqlite3`,
              GEIST_E2E_BACKEND_PORT: String(backendPort),
              GEIST_MEMORY_IDLE_SECONDS: '0',
            },
          },
        ]),
    {
      command: 'npm start',
      cwd: '.',
      url: `http://127.0.0.1:${frontendPort}/chat`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        ...process.env,
        BROWSER: 'none',
        HOST: '127.0.0.1',
        PORT: String(frontendPort),
        REACT_APP_BACKEND_HOST: '127.0.0.1',
        REACT_APP_BACKEND_PORT: String(backendPort),
      },
    },
  ],
});
