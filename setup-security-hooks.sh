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

# Install pre-commit
echo "📦 Installing pre-commit..."
pip install pre-commit

# Install other security tools
echo "📦 Installing security tools..."
pip install bandit safety black isort flake8 yamllint

# Install hadolint (Dockerfile linter)
echo "📦 Installing hadolint..."
if ! command -v hadolint &> /dev/null; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        wget -q https://github.com/hadolint/hadolint/releases/download/v2.13.1-beta/hadolint-Linux-x86_64 -O hadolint
        sudo mv hadolint /usr/local/bin/ || mv hadolint ~/bin/
        chmod +x /usr/local/bin/hadolint 2>/dev/null || chmod +x ~/bin/hadolint
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install hadolint
        else
            echo "⚠️  Please manually install hadolint from https://github.com/hadolint/hadolint/releases"
        fi
    fi
else
    echo "✓ Hadolint already installed"
fi

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
echo "   ✓ Dependency vulnerability scanning (safety)"
echo "   ✓ Code formatting (black, isort)"
echo "   ✓ Linting (flake8, yamllint)"
echo "   ✓ Dockerfile linting (hadolint)"
echo ""
