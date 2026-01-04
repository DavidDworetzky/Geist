# Pre-commit Hooks Setup

This document explains how to set up and use pre-commit hooks for Python type checking and code quality in the Geist project.

## Overview

Pre-commit hooks automatically run checks on your code before each commit. This project uses:
- **Ruff**: Fast Python linter and code formatter (replaces black, isort, flake8, and more)
- **mypy**: Python static type checking
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
pre-commit run ruff --all-files
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

### Ruff Configuration

Ruff combines linting and formatting in one fast tool. Configuration is in `pyproject.toml`:

**General settings:**
- Line length: 100 characters
- Target Python version: 3.11
- Excluded paths: migrations/, .venv/, build/, dist/, etc.

**Linting rules enabled:**
- `E/W`: pycodestyle errors and warnings
- `F`: Pyflakes (undefined names, unused imports)
- `I`: isort (import sorting)
- `N`: pep8-naming conventions
- `UP`: pyupgrade (modern Python syntax)
- `B`: flake8-bugbear (likely bugs and design problems)
- `C4`: flake8-comprehensions (better comprehensions)
- `SIM`: flake8-simplify (code simplification)

**Import sorting:**
- Known first-party modules: app, agents, adapters, utils
- Two blank lines after imports

**Formatting:**
- Double quotes for strings
- Space indentation
- Unix line endings (LF)

To customize rules, edit `[tool.ruff.lint]` in `pyproject.toml`

## Pre-commit Hook Details

### Ruff Linting
- Runs on: All `.py` files
- Purpose: Catches code quality issues, style violations, and potential bugs
- Action on failure: Auto-fixes many issues (unused imports, outdated syntax, etc.), you'll need to re-stage and commit
- Speed: Extremely fast (10-100x faster than traditional tools)

### Ruff Formatting
- Runs on: All `.py` files
- Purpose: Ensures consistent code style and formatting
- Action on failure: Auto-formats code, you'll need to re-stage and commit
- Compatible with: Black formatting style

### Type Checking (mypy)
- Runs on: All `.py` files (excluding tests/, scripts/, migrations/)
- Purpose: Catches type-related bugs before runtime through static analysis
- Action on failure: Blocks commit and shows type errors
- Note: Requires manual fixes (cannot auto-fix type errors)

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

### Ruff formatting conflicts

If Ruff's formatting conflicts with existing code style:
1. Run `ruff format .` to see what changes would be made
2. Review the changes
3. Adjust `[tool.ruff.format]` settings in `pyproject.toml` if needed

### Disabling specific Ruff rules

To disable a specific rule for a file, add to the top:
```python
# ruff: noqa: E501
```

To disable for a specific line:
```python
x = really_long_line()  # noqa: E501
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

## Why Ruff?

Ruff is a modern, fast alternative to multiple Python tools:
- **Speed**: Written in Rust, 10-100x faster than traditional Python linters
- **All-in-one**: Replaces black, isort, flake8, pyupgrade, and more
- **Compatible**: Follows black's formatting style by default
- **Actively maintained**: Regular updates and new features

## Additional Resources

- [pre-commit documentation](https://pre-commit.com/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [Ruff rules reference](https://docs.astral.sh/ruff/rules/)
