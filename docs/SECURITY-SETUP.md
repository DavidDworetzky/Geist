# Security Setup Guide

Quick start guide for setting up security checks in your Geist development environment.

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
./setup-security-hooks.sh
```

This script will:
- Install pre-commit framework
- Install all security tools (gitleaks, bandit, safety, hadolint)
- Configure git hooks
- Optionally run initial checks

### Option 2: Manual Setup

```bash
# Install pre-commit
pip install pre-commit

# Install security tools
pip install bandit safety black isort flake8 yamllint

# Install gitleaks (Linux)
wget https://github.com/gitleaks/gitleaks/releases/download/v8.21.2/gitleaks_8.21.2_linux_x64.tar.gz
tar -xzf gitleaks_8.21.2_linux_x64.tar.gz
sudo mv gitleaks /usr/local/bin/

# Install gitleaks (macOS)
brew install gitleaks

# Install hadolint (Linux)
wget https://github.com/hadolint/hadolint/releases/download/v2.13.1-beta/hadolint-Linux-x86_64 -O hadolint
sudo mv hadolint /usr/local/bin/
chmod +x /usr/local/bin/hadolint

# Install hadolint (macOS)
brew install hadolint

# Install pre-commit hooks
pre-commit install
```

## Verifying Installation

```bash
# Check pre-commit is installed
pre-commit --version

# Check gitleaks is installed
gitleaks version

# Check other tools
bandit --version
safety --version
hadolint --version
```

## Testing the Setup

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run specific security checks
pre-commit run gitleaks --all-files
pre-commit run bandit --all-files
pre-commit run python-safety-dependencies-check --all-files
```

## Daily Workflow

### Making a Commit

When you run `git commit`, pre-commit hooks will automatically run:

```bash
git add .
git commit -m "Your commit message"
# Hooks run automatically here
```

If hooks fail:
1. Review the output to understand what failed
2. Fix the issues
3. Stage the fixes with `git add`
4. Commit again

### Checking Before Push

```bash
# Run all checks before pushing
pre-commit run --all-files

# If all pass, push
git push
```

### Bypassing Hooks (Emergency Only)

```bash
# Skip hooks (not recommended)
git commit --no-verify

# Skip specific hook
SKIP=gitleaks git commit
```

## Common Issues and Solutions

### Hook Fails on First Run

Some hooks may fail on the first run after installation due to existing code:

```bash
# Fix formatting issues automatically
black .
isort .

# Re-run checks
pre-commit run --all-files
```

### Gitleaks False Positives

Add to `.gitleaks.toml` allowlist:

```toml
[allowlist]
regexes = [
    '''your-false-positive-pattern'''
]
```

### Bandit False Positives

Add inline comment:

```python
# This is safe because... [explanation]
some_code()  # nosec B601
```

Or configure in `pyproject.toml`:

```toml
[tool.bandit]
skips = ["B601"]
```

### Performance Issues

If hooks are slow on large commits:

```bash
# Run only on changed files
pre-commit run

# Skip slow checks temporarily
SKIP=safety git commit
```

## CI/CD Pipeline

The security checks also run automatically in GitHub Actions on:
- Every push to main branch
- Every pull request to main branch

### Viewing CI Results

1. Go to the "Actions" tab in GitHub
2. Click on your workflow run
3. Check the "security" and "container-security" jobs
4. Download security reports from artifacts

### Security Job Artifacts

After each CI run, download security reports:
- `gitleaks-report.json`: Secret scanning results
- `bandit-report.json`: Python SAST results
- `safety-report.json`: Dependency vulnerability report
- `npm-audit-report.json`: Frontend dependency report

## Updating Security Tools

### Update Pre-commit Hooks

```bash
# Update to latest versions
pre-commit autoupdate

# Install updated hooks
pre-commit install --install-hooks
```

### Update Security Tools

```bash
pip install --upgrade bandit safety black isort flake8 yamllint pre-commit
```

## Configuration Files

- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `.gitleaks.toml`: Gitleaks configuration and allowlists
- `.yamllint`: YAML linting rules
- `pyproject.toml`: Python tool configuration (bandit, black, isort)

## Need Help?

- Check the main [SECURITY.md](../SECURITY.md) for detailed documentation
- Review tool-specific documentation linked in SECURITY.md
- Open an issue or discussion on GitHub
- Contact the development team

## Best Practices

1. ✅ Run `pre-commit run --all-files` before pushing
2. ✅ Fix security issues immediately, don't bypass hooks
3. ✅ Keep security tools updated with `pre-commit autoupdate`
4. ✅ Review CI security reports regularly
5. ✅ Document any exceptions or bypasses
6. ❌ Don't use `--no-verify` unless absolutely necessary
7. ❌ Don't commit secrets or credentials
8. ❌ Don't ignore security warnings without investigation
