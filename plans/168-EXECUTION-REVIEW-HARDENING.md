# Execution review hardening

Carry the permissions parent into the execution layer without changing its base.
Require fresh approval for host or network access even with standing grants, apply
the same best-effort hardline check to both host-reaching backends, and validate
bind mounts before dispatch. Keep sandbox isolation distinct from approval.

Bound timeout cleanup for one-shot container and local process-group execution.
Expose policy refusal in the tool result, retain command exit status, and document
the limitations of regex checks, environment scrubbing, and custom images.

Verify with contract/registry tests, Docker backend tests, and harmless real local
timeout/cwd checks. No dependencies, user-data migrations, or new approval modes.
Interactive approval resume remains in #308; ship these layers together.

Follow-up: cap retained subprocess bytes at 64 KiB per stream and drain excess
without retaining it, preserving verbose commands and their exit status. Use the
same capture helper for Docker and local execution. Preserve drained timeout
output and reap children.
Expose mandatory-approval metadata in the catalog and disable new inert UI grants.
Reader threads own pipe closure to avoid descriptor-reuse races. Platform guards
must also type-check under the Windows CI target, without relying on os.name narrowing.
