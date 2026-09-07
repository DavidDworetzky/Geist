"""Local-only discovery for the pinned Qwen auxiliary drafter."""

import os
from pathlib import Path


DFLASH_REPO = "incoai/Qwen3.8-27B-DFlash2"
DFLASH_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"


def default_dflash_path() -> Path:
    from app.services.local_models import default_model_home

    return default_model_home() / "auxiliary" / "qwen3.8-27b-dflash2" / DFLASH_REVISION / "snapshot"


def find_dflash_path(model_id: str) -> Path | None:
    mode = os.environ.get("GEIST_MLX_DFLASH", "auto").casefold()
    if mode not in {"auto", "on", "off"}:
        raise ValueError("GEIST_MLX_DFLASH must be auto, on, or off")
    if mode == "off" or model_id.casefold() != "qwen/qwen3.8-27b":
        return None
    override = os.environ.get("GEIST_MLX_DFLASH_DIR")
    path = Path(override).expanduser() if override else default_dflash_path()
    if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
        return path
    if mode == "on" or override:
        raise FileNotFoundError(
            f"DFlash artifact missing at {path}; run scripts/download_mlx_dflash.py"
        )
    return None
