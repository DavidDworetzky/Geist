import { expect, test } from '@playwright/test';

test('stores and processes every file in a batch without closing the uploader', async ({ page }) => {
  await page.goto('/files');
  await page.getByRole('button', { name: 'Upload Files', exact: true }).click();
  const suffix = Date.now();
  const names = [`first-${suffix}.txt`, `second-${suffix}.txt`];
  await page.getByLabel('Choose files to upload').setInputFiles(names.map(name => ({
    name, mimeType: 'text/plain', buffer: Buffer.from(`Contents of ${name}`),
  })));
  for (const name of names) {
    const card = page.locator('.file-card').filter({ has: page.getByRole('heading', { name, exact: true }) });
    await expect(card.getByText('Processed', { exact: true })).toBeVisible();
  }
  await expect(page.getByRole('button', { name: 'Hide Upload' })).toBeVisible();
});
