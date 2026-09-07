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

Main integration retains authenticated workspace approval resume and dynamic
tool sources while preserving all execution hardline/capture/mandatory-approval
controls. Obsolete disabled email/SMS/Markdown-write registrations removed by
main stay removed. 475 Docker tests and218 frontend tests/build pass; project
mypy passes9 files. Normal merge hooks found only baseline cached-mypy Any in
unchanged whisper_adapter and Bandit's unchanged main all-interface bind. Those
two hooks alone are skipped for this merge; CI configuration is unchanged.
