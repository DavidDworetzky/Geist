---
title: Source Installation
description: Run Geist locally from its locked development environment.
order: 1
category: Getting Started
icon: download
---

# Source Installation

Public Pitchblend installers are still in development. Contributors and technical users can run the Geist engine from source.

## Install and start

From a Geist checkout:

```bash
make sync
make run
```

`make sync` creates the environment from the committed `uv.lock`. `make run` initializes the default local SQLite database and starts the backend.

## Optional local-model support

On Apple silicon, install the MLX extras when you need the native local-model runner:

```bash
uv sync --extra local-mlx
make run MLX_BACKEND=1
```

Linux users who need the Transformers runner can install its locked extra:

```bash
uv sync --extra local-transformers
```

## First model

After Geist starts, open its Models page. The available download or import flow depends on your platform and selected runner. Model downloads can be large, so verify that sufficient disk space is available before starting.

Do not add secrets directly to source-controlled configuration files. Use the supported local environment configuration for any provider or Hugging Face credentials.
