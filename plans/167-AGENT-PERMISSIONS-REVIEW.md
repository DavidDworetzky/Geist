# Agent permissions review follow-up

Keep the centralized approval gate and fail-safe default posture. Scope by-ID
settings reads/writes to the current principal, bound submitted and stored tool
allowlists, and only send permission changes from the UI when explicitly edited.
Expose allowlist selection state and disabled/high-impact tools accessibly.

This branch supplies the permissions model, not the interactive approval resume
loop. Ship it with #308 (and descendants) when exposing require-approval mode;
do not advertise #309 alone as a complete interactive approval experience.
The current-principal helper is still the pre-existing default-user placeholder,
not production authentication. Main's operator-auth integration is separate.

No dependency or schema changes. Verify owner rejection before service access,
allowlist limits/canonicalization, permission-save omission, and accessible
allowlist state with backend API/service tests and frontend tests.
