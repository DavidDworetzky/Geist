"""
Shared base for image generation adapters.

Concrete adapters (GeminiImageAdapter online, FluxImageAdapter offline)
implement _generate(prompt) -> list[bytes] and declare their generate_image
action with the @async_tool and @online_tool/@offline_tool decorators. The
base provides the common run template: generate bytes, persist artifacts
under the configured output directory, and return the result payload the
agent reads back through JobStatusAdapter.check_async_tool.
"""
import datetime
import logging
import os
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any

from adapters.base_adapter import BaseAdapter
from adapters.tool_modes import tool_mode


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "output/images"


class BaseImageGenAdapter(BaseAdapter):

    def __init__(self, **kwargs):
        self.output_dir = os.getenv("IMAGE_GEN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
        self.timeout = float(os.getenv("IMAGE_GEN_TIMEOUT_SECONDS", "120"))

    def enumerate_actions(self):
        return ["generate_image"]

    @abstractmethod
    def generate_image(self, prompt: str) -> dict[str, Any]:
        """Concrete adapters declare this with @async_tool and a mode decorator."""

    @abstractmethod
    def _generate(self, prompt: str) -> list[bytes]:
        """Produce raw image bytes for the prompt."""

    def _run(self, prompt: str) -> dict[str, Any]:
        """Common generate -> persist -> result template for generate_image."""
        images = self._generate(prompt)
        if not images:
            raise RuntimeError(f"{type(self).__name__} produced no images for prompt: {prompt!r}")
        return {
            "prompt": prompt,
            "mode": tool_mode(type(self).generate_image),
            "adapter": type(self).__name__,
            "file_paths": self._write_artifacts(images),
        }

    def _write_artifacts(self, images: list[bytes]) -> list[str]:
        """Write image bytes under output/images/<date>-<id>/ and return paths."""
        batch = f"{datetime.datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:8]}"
        directory = Path(self.output_dir) / batch
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, image_bytes in enumerate(images):
            path = directory / f"image_{index}.png"
            path.write_bytes(image_bytes)
            paths.append(str(path))
        logger.info(f"Wrote {len(paths)} generated image(s) to {directory}")
        return paths
