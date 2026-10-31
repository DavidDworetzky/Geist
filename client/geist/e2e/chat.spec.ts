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

test('streams through every local MLX chat layer before generation can finish', async ({ page }) => {
  const control = '/api/e2e/streaming';
  expect((await page.request.post(`${control}/start`)).ok()).toBe(true);
  try {
    await page.getByPlaceholder('Type your message...').fill('Exercise streaming backpressure');
    await page.getByRole('button', { name: 'Send' }).click();
    const answer = page.locator('.chat-message-ai');
    const responding = page.getByRole('status', { name: 'Geist is responding' });

    // Generation is blocked at a model-source gate, not a timer. Buffering in
    // any adapter, runner, agent, orchestrator, SSE hop, or UI makes this fail.
    await expect(answer).toContainText('STREAM-FIRST');
    await expect(answer).not.toContainText('STREAM-SECOND');
    await expect(responding).toBeVisible();
    expect(await (await page.request.get(`${control}/state`)).json()).toMatchObject({
      stage: 1, closed: false, tools_seen: true, released: [false, false],
    });

    expect((await page.request.post(`${control}/release/1`)).ok()).toBe(true);
    await expect(answer).toContainText('STREAM-FIRST STREAM-SECOND');
    await expect(answer).not.toContainText('STREAM-FINAL');
    await expect(responding).toBeVisible();
    expect(await (await page.request.get(`${control}/state`)).json()).toMatchObject({
      stage: 2, closed: false, released: [true, false],
    });

    expect((await page.request.post(`${control}/release/2`)).ok()).toBe(true);
    await expect(answer).toContainText('STREAM-FIRST STREAM-SECOND STREAM-FINAL');
    await expect(responding).toBeHidden();
    expect(await (await page.request.get(`${control}/state`)).json()).toMatchObject({
      stage: 3, closed: true,
    });
  } finally {
    // Also unblock the producer after a failed assertion or an aborted test.
    await page.request.post(`${control}/reset`);
  }
});

test('dispatches Qwen XML search calls and streams the tool-grounded answer', async ({ page }) => {
  const control = '/api/e2e/streaming';
  expect((await page.request.post(`${control}/start?scenario=xml_tool`)).ok()).toBe(true);
  try {
    await page.getByPlaceholder('Type your message...')
      .fill('Search the internet and find me some recent celebrity headlines.');
    await page.getByRole('button', { name: 'Send' }).click();
    const answer = page.locator('.chat-message-ai');
    await expect(answer).toContainText('STREAM-FIRST');
    await expect(page.getByRole('status', { name: 'Geist is responding' })).toBeVisible();
    expect(await (await page.request.get(`${control}/state`)).json()).toMatchObject({
      search_calls: [{ query: 'recent celebrity headlines', max_results: 3 }],
      tool_result_seen: true,
      closed: false,
    });
    await expect(answer).not.toContainText('<function=');
    expect((await page.request.post(`${control}/release/1`)).ok()).toBe(true);
    await expect(answer).toContainText('STREAM-FIRST STREAM-SECOND');
    expect((await page.request.post(`${control}/release/2`)).ok()).toBe(true);
    await expect(answer).toContainText('STREAM-FIRST STREAM-SECOND STREAM-FINAL');
    await expect(page.getByRole('status', { name: 'Geist is responding' })).toBeHidden();
  } finally {
    await page.request.post(`${control}/reset`);
  }
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

test('preserves streamed prose after a malformed tool failure and reload', async ({ page }) => {
  await page.getByPlaceholder('Type your message...').fill('Trigger failure after prose');
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.getByText(
    'Error: The model returned malformed tool calls and exhausted its recovery budget. Please try again.',
    { exact: true },
  )).toBeVisible();
  await expect(page.locator('.chat-message-ai')).toContainText('Working on it.');
  await expect(page).toHaveURL(/\/chat\/\d+$/);
  await page.reload();
  await expect(page.locator('.chat-message-ai')).toContainText('Working on it.');
  await expect(page.getByText('Turn status: failed', { exact: true })).toBeVisible();
  await expect(page.locator('body')).not.toContainText('<tool_call>');
});

test('recovers malformed tool output and persists the corrected answer', async ({ page }) => {
  await page.getByPlaceholder('Type your message...').fill('Trigger recoverable malformed output');
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.locator('.chat-message-ai')).toContainText('Recovered after retry.');
  await expect(page.getByRole('status', { name: 'Geist is responding' })).toBeHidden();
  await expect(page.getByText(backendFailureMessage, { exact: true })).toBeHidden();
  await page.reload();
  await expect(page.locator('.chat-message-ai')).toContainText('Recovered after retry.');
});
