import { expect, test } from '@playwright/test';

test('intent routing defaults off and persists an explicit opt-in and opt-out', async ({ page }) => {
  await page.goto('/settings');
  const toggle = page.getByRole('button', { name: 'Intent Router' });
  await expect(toggle).toHaveAttribute('aria-pressed', 'false');
  try {
    await toggle.click();
    await page.getByRole('button', { name: 'Save Changes' }).click();
    await expect(page.getByText('Settings saved successfully.', { exact: true })).toBeVisible();
    await page.reload();
    await expect(toggle).toHaveAttribute('aria-pressed', 'true');
  } finally {
    // Restore the shared test workspace even when persistence assertions fail.
    await page.reload();
    if (await toggle.getAttribute('aria-pressed') === 'true') {
      await toggle.click();
      await page.getByRole('button', { name: 'Save Changes' }).click();
      await expect(page.getByText('Settings saved successfully.', { exact: true })).toBeVisible();
    }
  }
  await page.reload();
  await expect(toggle).toHaveAttribute('aria-pressed', 'false');
});
