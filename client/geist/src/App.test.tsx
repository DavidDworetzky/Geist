import React from 'react';
import { render, screen, within } from '@testing-library/react';
import App from './App';

test('renders the requested sidebar navigation in order', () => {
  render(<App />);

  const sidebar = screen.getByRole('complementary', { name: /primary/i });
  const navigation = within(sidebar).getByRole('navigation');
  const navigationLinks = within(navigation).getAllByRole('link');

  expect(within(sidebar).getByText(/Geist/i)).toBeInTheDocument();
  expect(navigationLinks).toHaveLength(5);
  expect(navigationLinks[0]).toHaveAccessibleName(/Chat/);
  expect(navigationLinks[1]).toHaveAccessibleName(/Models/);
  expect(navigationLinks[2]).toHaveAccessibleName(/Settings/);
  expect(navigationLinks[3]).toHaveAccessibleName(/Tools/);
  expect(navigationLinks[4]).toHaveAccessibleName(/Files/);
  expect(within(navigation).queryByText(/Workflows/i)).not.toBeInTheDocument();
});
