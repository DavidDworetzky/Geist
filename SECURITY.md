# Security Policy

## Overview

Geist implements comprehensive security checks through both pre-commit hooks and CI/CD pipeline automation to ensure code quality and prevent security vulnerabilities.

## Security Measures

### 1. Pre-commit Hooks

Pre-commit hooks run automatically before each commit to catch security issues early in development.

#### Installation

Run the setup script to install all security tools and hooks:

```bash
./setup-security-hooks.sh
```

Or install manually:

```bash
pip install pre-commit
pre-commit install
```

#### Included Checks

- **Secret Detection** (gitleaks): Scans for accidentally committed secrets, API keys, passwords
- **Python SAST** (bandit): Static analysis security testing for Python code
- **Dependency Scanning** (safety): Checks Python dependencies for known vulnerabilities
- **Code Formatting** (black, isort): Ensures consistent code style
- **Linting** (flake8, yamllint): Code quality and syntax checking
- **Dockerfile Linting** (hadolint): Best practices for Dockerfiles
- **General Checks**: Trailing whitespace, merge conflicts, large files, private keys

#### Running Pre-commit Hooks

Hooks run automatically on commit. To run manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run bandit
pre-commit run gitleaks
```

#### Bypassing Hooks (Not Recommended)

Only in exceptional circumstances:

```bash
git commit --no-verify
```

**Warning:** Bypassing hooks may allow security vulnerabilities to be committed.

### 2. CI/CD Security Pipeline

The CI/CD pipeline includes three security-focused jobs that run on every push and pull request:

#### Security Job

Runs comprehensive security scans:

- **Gitleaks**: Full repository secret scanning with history analysis
- **Bandit**: Python SAST with severity-based failure (fails on HIGH severity)
- **Safety**: Python dependency vulnerability scanning
- **NPM Audit**: Frontend dependency vulnerability scanning (fails on CRITICAL vulnerabilities)

All security reports are uploaded as artifacts for review.

#### Container Security Job

- **Trivy**: Scans Dockerfiles and configurations for security issues
- Results uploaded to GitHub Security tab (SARIF format)

#### Build Job with Image Scanning

After successful builds:

- **Trivy**: Scans built Docker images for vulnerabilities
- Checks for CRITICAL and HIGH severity issues

### 3. Configuration Files

- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `.gitleaks.toml`: Gitleaks secret detection rules and allowlists
- `.yamllint`: YAML linting rules
- `pyproject.toml`: Bandit, black, and isort configuration

## Severity Levels and Failure Conditions

### Pre-commit Hooks

- **BLOCK**: Secret detection, merge conflicts, syntax errors
- **WARN**: Code formatting, minor linting issues

### CI/CD Pipeline

- **FAIL on**: HIGH severity security issues (Bandit), CRITICAL npm vulnerabilities, detected secrets
- **REPORT**: All other findings are reported but don't fail the build

## Handling Security Findings

### False Positives

If a security tool reports a false positive:

1. **Gitleaks**: Add pattern to `.gitleaks.toml` allowlist
2. **Bandit**: Add `# nosec` comment with justification or configure in `pyproject.toml`
3. **Safety**: Pin dependency version if upgrade not possible, document reason

### True Positives

1. **Secrets**: Remove from git history using `git-filter-repo` or BFG, rotate credentials
2. **Vulnerabilities**: Update dependencies, apply patches, or implement workarounds
3. **Code Issues**: Refactor code to address security concerns

## Security Best Practices

1. **Never commit secrets**: Use environment variables or secret management tools
2. **Keep dependencies updated**: Regularly run `npm audit fix` and update Python packages
3. **Review security reports**: Check CI/CD artifacts for detailed findings
4. **Run local checks**: Test pre-commit hooks before pushing
5. **Document exceptions**: If bypassing a check, document why in commit message

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Geist:

1. **Do not** open a public GitHub issue
2. Contact the maintainers privately
3. Provide detailed information about the vulnerability
4. Allow time for a fix before public disclosure

## Security Tools Documentation

- [Gitleaks](https://github.com/gitleaks/gitleaks) - Secret scanning
- [Bandit](https://bandit.readthedocs.io/) - Python SAST
- [Safety](https://pyup.io/safety/) - Dependency scanning
- [Trivy](https://trivy.dev/) - Container scanning
- [Hadolint](https://github.com/hadolint/hadolint) - Dockerfile linting
- [Pre-commit](https://pre-commit.com/) - Git hooks framework

## Maintenance

### Updating Security Tools

Update pre-commit hooks:

```bash
pre-commit autoupdate
```

Update security tools:

```bash
pip install --upgrade bandit safety pre-commit
```

### Security Scan Schedule

- **Pre-commit**: On every commit
- **CI/CD**: On every push and pull request
- **Manual scans**: Run as needed during development

## Questions?

For questions about security tooling or practices, please open a GitHub discussion or contact the maintainers.
