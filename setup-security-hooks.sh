#!/bin/bash
# Setup script for installing security pre-commit hooks
# Run this script after cloning the repository

set -e

echo "🔒 Setting up security pre-commit hooks for Geist..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.11+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Install only the pinned hook runner. Pre-commit provisions Bandit, Yamllint,
# and Hadolint from the pinned revisions in .pre-commit-config.yaml.
echo "📦 Installing pre-commit..."
python3 -m pip install --only-binary=:all: pre-commit==4.0.1

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files (optional, for initial setup)
read -p "Do you want to run pre-commit checks on all existing files? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔍 Running pre-commit on all files..."
    pre-commit run --all-files || echo "⚠️  Some checks failed. Please review and fix the issues."
fi

echo ""
echo "✅ Security hooks setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Pre-commit hooks will now run automatically on git commit"
echo "   2. To run hooks manually: pre-commit run --all-files"
echo "   3. To bypass hooks (not recommended): git commit --no-verify"
echo "   4. Review .pre-commit-config.yaml to customize checks"
echo ""
echo "🔒 Security checks enabled:"
echo "   ✓ Python security scanning (bandit)"
echo "   ✓ Secret scanning (staged-secret-scan)"
echo "   ✓ Code formatting and linting (ruff, yamllint)"
echo "   ✓ Dockerfile linting (hadolint)"
echo ""
