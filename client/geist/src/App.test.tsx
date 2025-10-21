import React from 'react';
import { render, screen, within } from '@testing-library/react';
import App from './App';

test('renders navigation and routes', () => {
  render(<App />);

  // Get the header/navigation section
  const header = screen.getByRole('complementary');

  // Check for navigation links within the header
  expect(within(header).getByText(/Home/i)).toBeInTheDocument();
  expect(within(header).getByText(/Chat/i)).toBeInTheDocument();
  expect(within(header).getByText(/Workflows/i)).toBeInTheDocument();
  expect(within(header).getByText(/Files/i)).toBeInTheDocument();
  expect(within(header).getByText(/Settings/i)).toBeInTheDocument();
});
