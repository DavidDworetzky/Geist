#!/usr/bin/env python3
"""
Script to process conda environment files and remove build fingerprints.

This script provides functionality to:
1. Copy linux_environment.yml to linux_environment_x86_x64.yml
2. Remove fingerprint components (the last =ZZZZZ part) from conda dependencies
3. Keep pip dependencies unchanged

Example transformation:
  _openmp_mutex=4.5=2_gnu  ->  _openmp_mutex=4.5
  libblas=3.9.0=26_linuxaarch64_openblas  ->  libblas=3.9.0
"""

import os
import logging
from pathlib import Path
from typing import Tuple
import re

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def remove_fingerprint(line: str) -> str:
    """
    Remove the fingerprint component from a conda dependency line.

    Conda dependencies follow the pattern: package=version=build_string
    This function removes the build_string (fingerprint) component.

    Args:
        line: A line from the conda environment file

    Returns:
        The line with the fingerprint removed if applicable, otherwise unchanged
    """
    # Strip whitespace to work with the actual content
    stripped = line.strip()
    leading_spaces = line[:len(line) - len(line.lstrip())]

    # Skip lines that don't contain dependencies
    # Pip dependencies use == instead of =
    if '==' in stripped:
        return line

    if '=' not in stripped:
        return line

    # Skip special lines (comments, channels, etc.)
    if stripped.startswith('#') or ':' in stripped:
        return line

    # Check if this is a conda dependency line (starts with - )
    if not stripped.startswith('-'):
        return line

    # Remove the leading '- ' to work with the package spec
    package_spec = stripped[2:].strip()

    # Match pattern: package=version=build_string
    # We want to remove the last =build_string part
    parts = package_spec.split('=')
    if len(parts) >= 3:
        # Keep only package and version, remove build string
        processed = f"{parts[0]}={parts[1]}"
        return leading_spaces + '- ' + processed + '\n'

    return line


def copy_environment(source_path: Path = None,
                     dest_path: Path = None) -> Tuple[bool, str]:
    """
    Copy environment file and remove fingerprints from conda dependencies.

    Args:
        source_path: Path to source environment file (defaults to linux_environment.yml)
        dest_path: Path to destination file (defaults to linux_environment_x86_x64.yml)

    Returns:
        Tuple containing:
            - Success status (bool)
            - Message describing the result
    """
    # Get the project root directory (assuming this script is in scripts/)
    root_dir = Path(__file__).parent.parent

    if source_path is None:
        source_path = root_dir / "linux_environment.yml"

    if dest_path is None:
        dest_path = root_dir / "linux_environment_x86_x64.yml"

    source_path = Path(source_path)
    dest_path = Path(dest_path)

    if not source_path.exists():
        logger.error(f"Source file {source_path} does not exist.")
        return False, f"Source file {source_path} does not exist."

    try:
        # Read source file
        with open(source_path, 'r') as f:
            lines = f.readlines()

        # Process each line
        processed_lines = []
        in_pip_section = False
        conda_deps_processed = 0

        for line in lines:
            # Track if we're in the pip section
            if '- pip:' in line:
                in_pip_section = True
                processed_lines.append(line)
                continue

            # If in pip section, don't process (pip uses == not =)
            if in_pip_section:
                processed_lines.append(line)
                continue

            # Process conda dependencies
            original_line = line
            processed_line = remove_fingerprint(line)

            if original_line != processed_line:
                conda_deps_processed += 1
                logger.debug(f"Processed: {original_line.strip()} -> {processed_line.strip()}")

            processed_lines.append(processed_line)

        # Write to destination file
        with open(dest_path, 'w') as f:
            f.writelines(processed_lines)

        logger.info(f"Successfully processed {conda_deps_processed} conda dependencies.")
        logger.info(f"Output written to {dest_path}")
        return True, f"Successfully copied and processed environment file. Modified {conda_deps_processed} conda dependencies."

    except Exception as e:
        logger.error(f"Error processing environment file: {str(e)}")
        return False, f"Error processing environment file: {str(e)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process conda environment file and remove build fingerprints"
    )

    parser.add_argument(
        "command",
        choices=["copy"],
        help="Command to execute"
    )

    parser.add_argument(
        "--source",
        type=str,
        help="Source environment file (defaults to linux_environment.yml)"
    )

    parser.add_argument(
        "--dest",
        type=str,
        help="Destination environment file (defaults to linux_environment_x86_x64.yml)"
    )

    args = parser.parse_args()

    if args.command == "copy":
        source = Path(args.source) if args.source else None
        dest = Path(args.dest) if args.dest else None
        success, message = copy_environment(source, dest)
        print(message)
        exit(0 if success else 1)

    else:
        parser.print_help()
