import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders navigation and routes', () => {
  render(<App />);
  expect(screen.getByText(/Home/i)).toBeInTheDocument();
  expect(screen.getByText(/Chat/i)).toBeInTheDocument();
  expect(screen.getByText(/Workflows/i)).toBeInTheDocument();
  expect(screen.getByText(/Files/i)).toBeInTheDocument();
  expect(screen.getByText(/Settings/i)).toBeInTheDocument();
});
