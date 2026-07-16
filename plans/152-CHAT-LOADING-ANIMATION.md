# Chat Loading Animation

## Goal

Show a Geist-themed animated ellipsis at the end of the chat history while the assistant response request is in progress.

## Implementation

1. Pass the existing chat request loading state into `ChatTextArea`.
2. Render an accessible three-dot status indicator after the current messages.
3. Style the indicator with the app's surface and accent tokens, staggered dot motion, and a reduced-motion fallback.
4. Add focused component coverage for the loading and idle states.

## Verification

1. Run the focused frontend test.
2. Run the production frontend build.
3. Start the application with Docker, inspect container logs, and verify the UI at `localhost:3000` when the local environment supports it.
