#!/usr/bin/env python3
"""Explicitly install the immutable auxiliary drafter (approximately 3.85 GB)."""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download

from agents.architectures.llama.dflash_artifact import (
    DFLASH_REPO,
    DFLASH_REVISION,
    default_dflash_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path)
    args = parser.parse_args()
    destination = args.weights_dir or default_dflash_path()
    snapshot_download(
        repo_id=DFLASH_REPO,
        revision=DFLASH_REVISION,
        local_dir=destination,
        allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE*"],
    )
    print(f"Installed {DFLASH_REPO}@{DFLASH_REVISION} at {destination}")


if __name__ == "__main__":
    main()
