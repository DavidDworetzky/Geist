---
title: Troubleshooting
description: Resolve common startup, model, and connection problems.
order: 4
category: Help
icon: help-circle
---

# Troubleshooting

## The application does not start

Run the locked environment setup again and review the terminal output:

```bash
make sync
make run
```

Check that another process is not already using the configured application ports.

## A local model will not load

- Confirm that the selected runner supports your operating system and architecture.
- Verify that the model download completed and that enough disk space remains.
- For gated Hugging Face models, confirm that you accepted the model license and configured an authorized token.
- On Apple silicon, confirm that the MLX extras are installed before using the MLX backend.

## The browser cannot reach Geist

Confirm that the backend is still running and use the URL printed by the startup command. If the frontend is running separately, confirm that its API base URL points to the Geist backend.

## Reporting a problem

Include your operating system, hardware architecture, startup command, selected runner, and the smallest relevant error excerpt. Never include access tokens, provider keys, or local environment files in a report.
