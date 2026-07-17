import { expect, Page, test } from '@playwright/test';

const backendFailureMessage =
  'Error: Chat backend failed to start. Check the configured model, local weights, and required credentials.';
const consoleErrorsByPage = new WeakMap<Page, string[]>();

test.beforeEach(async ({ page }) => {
  const consoleErrors: string[] = [];
  consoleErrorsByPage.set(page, consoleErrors);
  page.on('console', (message) => {
    if (message.type() === 'error') {
      consoleErrors.push(message.text());
    }
  });
  await page.goto('/chat');
});

test.afterEach(async ({ page }) => {
  expect(consoleErrorsByPage.get(page) ?? []).toEqual([]);
});

test('completes a chat through the real SSE route', async ({ page }) => {
  const messageInput = page.getByPlaceholder('Type your message...');
  await messageInput.fill('Reply for E2E');
  await page.getByRole('button', { name: 'Send' }).click();

  await expect(page.locator('.chat-message-ai')).toContainText('E2E chat works.');
  await expect(page.getByRole('status', { name: 'Geist is responding' })).toBeHidden();
  await expect(messageInput).toBeEnabled();
});

test('leaves connecting and surfaces a safe model startup error', async ({ page }) => {
  const messageInput = page.getByPlaceholder('Type your message...');
  await messageInput.fill('Trigger backend failure');
  await page.getByRole('button', { name: 'Send' }).click();

  await expect(page.getByText(backendFailureMessage, { exact: true })).toBeVisible();
  await expect(page.getByText('Turn status: failed', { exact: true })).toBeVisible();
  await expect(page.getByRole('status', { name: 'Geist is responding' })).toBeHidden();
  await expect(messageInput).toBeEnabled();
});
