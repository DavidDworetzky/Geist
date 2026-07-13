import React from 'react';
import { render, screen, within } from '@testing-library/react';
import App from './App';

test('renders sidebar navigation and routes', () => {
  render(<App />);

  const sidebar = screen.getByRole('complementary', { name: /primary/i });

  expect(within(sidebar).getByText(/Geist/i)).toBeInTheDocument();
  expect(within(sidebar).getByText(/Chat/i)).toBeInTheDocument();
  expect(within(sidebar).getByText(/Workflows/i)).toBeInTheDocument();
  expect(within(sidebar).getByText(/Files/i)).toBeInTheDocument();
  expect(within(sidebar).getByText(/Models/i)).toBeInTheDocument();
  expect(within(sidebar).getByText(/Settings/i)).toBeInTheDocument();
});
