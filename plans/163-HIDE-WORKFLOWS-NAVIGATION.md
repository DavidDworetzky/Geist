# Hide Workflows Navigation

## Goal

Hide the Workflows tab from the primary UI navigation and display the remaining tabs in this order: Chat, Models, Settings, Files.

## Implementation

1. Remove the Workflows item from the `AppShell` navigation configuration.
2. Reorder the remaining navigation items without removing the existing workflow routes.
3. Update the app navigation test to verify both the requested order and the absence of Workflows.

## Verification

- Run the focused `App.test.tsx` frontend test.
- Run the frontend production build to catch TypeScript or bundling regressions.
