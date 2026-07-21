import { expect, test } from '@playwright/test';


async function sendMemoryChat(
  page: import('@playwright/test').Page,
  message: string,
  expectedStatus = 'ready',
) {
  await page.getByPlaceholder('Type your message...').fill(message);
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page).toHaveURL(/\/chat\/\d+$/);
  const match = page.url().match(/\/chat\/(\d+)$/);
  if (!match) throw new Error('Chat id was not assigned');
  const chatId = Number(match[1]);
  await expect.poll(async () => {
    const response = await page.request.get(`/api/v1/memory/chats/${chatId}`);
    if (!response.ok()) return 'missing';
    return (await response.json()).status;
  }).toBe(expectedStatus);
  return chatId;
}


async function search(
  page: import('@playwright/test').Page,
  payload: Record<string, unknown>,
) {
  const response = await page.request.post('/api/v1/memory/search', { data: payload });
  expect(response.ok()).toBeTruthy();
  return (await response.json()).results as Array<{ content: string }>;
}


test.beforeEach(async ({ page }) => {
  await page.goto('/chat');
});


test('public chat promotes an explicit durable fact to global memory', async ({ page }) => {
  await sendMemoryChat(page, 'Remember ultramarine.');

  const globalResults = await search(page, {
    query: 'ultramarine',
    scope: 'user',
  });
  expect(globalResults.some(result => result.content.includes('ultramarine'))).toBeTruthy();

  await page.getByRole('button', { name: 'Expand chat history' }).click();
  await page.getByText('Profile memory').click();
  await expect(
    page.locator('.memory-explorer').getByText(
      'The user asked Geist to remember: ultramarine',
      { exact: true },
    ),
  ).toBeVisible();
});


test('private chat is searchable only inside its own thread', async ({ page }) => {
  await page.getByRole('switch', { name: 'Private' }).click();
  await expect(page.getByText('Stored only in this private chat')).toBeVisible();
  const chatId = await sendMemoryChat(page, 'Remember obsidian-e2e.');

  const globalResults = await search(page, {
    query: 'obsidian-e2e',
    scope: 'user',
  });
  expect(globalResults.some(result => result.content.includes('obsidian-e2e'))).toBeFalsy();

  const threadResults = await search(page, {
    query: 'obsidian-e2e',
    scope: 'thread',
    chat_session_id: chatId,
  });
  expect(threadResults.some(result => result.content.includes('obsidian-e2e'))).toBeTruthy();
});


test('memory disabled prevents chat-derived storage', async ({ page }) => {
  await page.getByRole('switch', { name: 'Memory enabled' }).click();
  await expect(page.getByText('Memory is off for this chat')).toBeVisible();
  await sendMemoryChat(page, 'Remember vermilion-e2e.', 'disabled');

  const globalResults = await search(page, {
    query: 'vermilion-e2e',
    scope: 'user',
  });
  expect(globalResults.some(result => result.content.includes('vermilion-e2e'))).toBeFalsy();
});


test('folder memory stays inside the selected private folder', async ({ page }) => {
  await page.getByRole('button', { name: 'Expand chat history' }).click();
  await page.getByRole('button', { name: 'New private folder' }).click();
  await page.getByRole('textbox', { name: 'Folder name' }).fill('E2E Vault');
  await page.getByRole('button', { name: 'Save folder' }).click();
  await expect(
    page.getByRole('button', { name: /^E2E Vault \d+ chats? · Private$/ }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Close chat history' }).click();

  const folderResponse = await page.request.get('/api/v1/memory/folders');
  const folders = await folderResponse.json();
  const folder = folders.find((item: { name: string }) => item.name === 'E2E Vault');
  expect(folder).toBeTruthy();
  await page.getByRole('combobox', { name: 'Memory folder' }).selectOption(
    String(folder.folder_id),
  );
  await expect(page.getByText('Stored only in this private folder')).toBeVisible();
  const chatId = await sendMemoryChat(page, 'Remember topaz-e2e launches Friday.');

  const folderResults = await search(page, {
    query: 'topaz-e2e',
    scope: 'folder',
    folder_id: folder.folder_id,
  });
  expect(folderResults.some(result => result.content.includes('topaz-e2e'))).toBeTruthy();

  await page.getByRole('button', { name: 'Expand chat history' }).click();
  await page.getByText('Folder memory').last().click();
  await page.getByRole('textbox', { name: 'Search memory' }).fill('topaz-e2e');
  await page.getByRole('button', { name: 'Run memory search' }).click();
  await expect(page.getByText(/topaz-e2e/).last()).toBeVisible();

  const otherFolderResponse = await page.request.post('/api/v1/memory/folders', {
    data: { name: 'E2E Other Vault', color: 'slate' },
  });
  const otherFolder = await otherFolderResponse.json();
  const otherResults = await search(page, {
    query: 'topaz-e2e',
    scope: 'folder',
    folder_id: otherFolder.folder_id,
  });
  expect(otherResults).toEqual([]);

  const globalResults = await search(page, {
    query: 'topaz-e2e',
    scope: 'user',
  });
  expect(globalResults.some(result => result.content.includes('topaz-e2e'))).toBeFalsy();

  await page.getByRole('button', { name: 'Rename E2E Vault folder' }).click();
  await page.getByRole('textbox', { name: 'Rename folder' }).fill('E2E Renamed Vault');
  await page.getByRole('button', { name: 'Save folder name' }).click();
  await expect(
    page.getByRole('button', { name: /^E2E Renamed Vault \d+ chats? · Private$/ }),
  ).toBeVisible();

  await page.getByRole('button', { name: 'Delete E2E Renamed Vault folder' }).click();
  await expect(page.getByText('Chats stay private and become unfiled.')).toBeVisible();
  await page.getByRole('button', { name: 'Delete folder', exact: true }).click();
  await expect(
    page.getByRole('button', { name: /^E2E Renamed Vault \d+ chats? · Private$/ }),
  ).toBeHidden();
  const settings = await (await page.request.get(`/api/v1/memory/chats/${chatId}`)).json();
  expect(settings.folder_id).toBeNull();
  expect(settings.effective_scope).toBe('private');
});
