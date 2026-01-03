# Pre-commit Hooks Setup

This document explains how to set up and use pre-commit hooks for Python type checking in the Geist project.

## Overview

Pre-commit hooks automatically run checks on your code before each commit. This project uses:
- **mypy**: Python type checking
- **black**: Code formatting
- **isort**: Import sorting
- **Standard pre-commit hooks**: File checks, YAML validation, etc.

## Installation

### 1. Update your conda environment

If you haven't already updated your conda environment with the latest dependencies:

```bash
# For Linux (ARM)
conda env update -f linux_environment.yml

# For Linux (x86_64)
conda env update -f linux_environment_x86_x64.yml

# For macOS (ARM)
conda env update -f mac_environment_arm.yml
```

### 2. Install pre-commit hooks

After updating your environment, install the pre-commit hooks:

```bash
pre-commit install
```

This command sets up git hooks that will run automatically before each commit.

## Usage

### Automatic execution

Once installed, the hooks will run automatically when you try to commit:

```bash
git add .
git commit -m "Your commit message"
```

If any checks fail, the commit will be blocked and you'll see error messages.

### Manual execution

You can run the hooks manually on all files:

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file.py

# Run a specific hook
pre-commit run mypy --all-files
```

### Bypassing hooks (not recommended)

In rare cases where you need to commit without running hooks:

```bash
git commit -m "Your message" --no-verify
```

**Note**: This should be avoided in normal workflow.

## Configuration

### mypy Configuration

Type checking configuration is in `pyproject.toml`. Key settings:

- **Python version**: 3.11
- **Excluded paths**: tests/, scripts/, migrations/
- **Strictness**: Currently set to moderate (can be increased later)

To adjust strictness, edit these settings in `pyproject.toml`:
```toml
[tool.mypy]
disallow_untyped_defs = true  # Make stricter
disallow_untyped_calls = true  # Make stricter
```

### black Configuration

Code formatting settings in `pyproject.toml`:
- Line length: 100 characters
- Target Python version: 3.11

### isort Configuration

Import sorting settings in `pyproject.toml`:
- Profile: black (compatible with black formatter)
- Line length: 100 characters
- Known first-party modules: app, agents, adapters, utils

## Pre-commit Hook Details

### Type Checking (mypy)
- Runs on: All `.py` files (excluding tests/, scripts/, migrations/)
- Purpose: Catches type-related bugs before runtime
- Action on failure: Blocks commit and shows type errors

### Code Formatting (black)
- Runs on: All `.py` files
- Purpose: Ensures consistent code style
- Action on failure: Auto-formats code, you'll need to re-stage and commit

### Import Sorting (isort)
- Runs on: All `.py` files
- Purpose: Organizes imports consistently
- Action on failure: Auto-sorts imports, you'll need to re-stage and commit

### Other Checks
- Large file detection (max 1MB)
- Merge conflict detection
- YAML syntax validation (excluding conda environment files)
- TOML syntax validation
- Private key detection
- Trailing whitespace removal
- End-of-file newline enforcement

## Troubleshooting

### "command not found: pre-commit"

Make sure you've activated the conda environment:
```bash
conda activate geist-linux-docker  # or your environment name
```

### Type checking errors on third-party libraries

If mypy complains about missing type stubs, you can:
1. Install type stubs: `pip install types-<library-name>`
2. Or add to `pyproject.toml`:
   ```toml
   [[tool.mypy.overrides]]
   module = "library_name.*"
   ignore_missing_imports = true
   ```

### Hooks are too slow

You can temporarily skip hooks for a commit:
```bash
git commit -m "Your message" --no-verify
```

But remember to run them before pushing:
```bash
pre-commit run --all-files
```

### Updating hooks

To update pre-commit hooks to their latest versions:
```bash
pre-commit autoupdate
```

## Best Practices

1. **Run hooks locally**: Don't rely only on CI/CD. Run `pre-commit run --all-files` before pushing.

2. **Fix issues incrementally**: If you have many type errors, fix them gradually by:
   - Adding `# type: ignore` comments temporarily
   - Creating a plan to address them over time
   - Gradually increasing mypy strictness

3. **Keep configuration updated**: Review and update `pyproject.toml` settings as the project matures.

4. **Use type hints**: Add type hints to new code to maximize the benefits of type checking.

## Additional Resources

- [pre-commit documentation](https://pre-commit.com/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [black documentation](https://black.readthedocs.io/)
- [isort documentation](https://pycqa.github.io/isort/)
