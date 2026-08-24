import React from 'react';
import { render, screen, within } from '@testing-library/react';
import App from './App';

test('renders the requested sidebar navigation in order', () => {
  render(<App />);

  const sidebar = screen.getByRole('complementary', { name: /primary/i });
  const navigation = within(sidebar).getByRole('navigation');
  const navigationLabels = within(navigation)
    .getAllByRole('link')
    .map((link) => link.querySelector('.nav-label')?.textContent);

  expect(within(sidebar).getByText(/Geist/i)).toBeInTheDocument();
  expect(navigationLabels).toEqual(['Chat', 'Models', 'Settings', 'Files']);
  expect(within(navigation).queryByText(/Workflows/i)).not.toBeInTheDocument();
});
