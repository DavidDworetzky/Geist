# Runtime Local Model Dropdown

## Goal

Replace the read-only local-model pill in the upper-right runtime summary with
an accessible dropdown containing the runnable local model artifacts installed
on this computer. Selecting an entry should immediately activate its model and
artifact.

## Scope

1. Load managed local artifacts from `/api/v1/models/local/artifacts` and show
   only supported entries whose status is `installed`.
2. Render the active local artifact as a native select styled like the existing
   runtime pill. Keep the read-only provider/model pills for online runtimes.
3. Persist selections through `useUserSettings().updateSettings`, including the
   model ID and concrete artifact ID required by the local runner.
4. Preserve the configured model as disabled fallback text when no matching
   installed artifact is available.
5. Cover filtering, selection, failed-save rollback, and online runtime display
   with focused AppShell tests.

## Verification

- Run the focused AppShell frontend test suite.
- Run the complete frontend suite and production build.
- Follow the Geist test loop for Docker and browser smoke verification where
  the environment supports it.
