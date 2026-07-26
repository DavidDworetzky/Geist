"""Minimal PyInstaller entry point for the Geist CLI."""

from app.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
