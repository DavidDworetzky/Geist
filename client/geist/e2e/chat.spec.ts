import { expect, Page, test } from '@playwright/test';

const backendFailureMessage =
  'Error: Chat completion failed';
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

test('persists a conversation and hydrates structured follow-up context', async ({ page }) => {
  const messageInput = page.getByPlaceholder('Type your message...');
  await messageInput.fill('Remember cobalt.');
  await page.getByRole('button', { name: 'Send' }).click();

  await expect(page.locator('.chat-message-ai')).toContainText('I will remember cobalt.');
  await expect(page).toHaveURL(/\/chat\/\d+$/);

  await messageInput.fill('What should you remember?');
  await page.getByRole('button', { name: 'Send' }).click();
  const followUpTurn = page.locator('.chat-turn').filter({
    hasText: 'What should you remember?',
  });
  await expect(followUpTurn.locator('.chat-message-ai')).toContainText('cobalt');

  await page.reload();
  await expect(
    page.locator('.chat-message-user').filter({ hasText: 'Remember cobalt.' }),
  ).toHaveCount(1);
  await expect(
    page.locator('.chat-message-user').filter({ hasText: 'What should you remember?' }),
  ).toHaveCount(1);
  await expect(followUpTurn.locator('.chat-message-ai')).toContainText('cobalt');
});

test('leaves connecting and surfaces a safe model failure', async ({ page }) => {
  const messageInput = page.getByPlaceholder('Type your message...');
  await messageInput.fill('Trigger backend failure');
  await page.getByRole('button', { name: 'Send' }).click();

  await expect(page.getByText(backendFailureMessage, { exact: true })).toBeVisible();
  await expect(page.getByText('Turn status: failed', { exact: true })).toBeVisible();
  await expect(page.getByRole('status', { name: 'Geist is responding' })).toBeHidden();
  await expect(messageInput).toBeEnabled();
});
