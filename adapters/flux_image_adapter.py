"""
Offline (local) image generation, defaulting to a FLUX.1-schnell backend.

generate_image is an @async_tool: the agent gets a job handle immediately
and polls JobStatusAdapter.check_async_tool(job_id=...) for the artifact
paths. The heavy diffusion runtime is not bundled with the default install:
IMAGE_GEN_OFFLINE_BACKEND names a module exposing
generate(prompt) -> list[bytes] (e.g. a FLUX.1-schnell shim), which is
imported lazily inside the worker.
"""

import importlib
import logging
import os
from typing import Any

from adapters.async_tool import async_tool
from adapters.image_gen_base import BaseImageGenAdapter
from adapters.tool_modes import offline_tool


logger = logging.getLogger(__name__)


class FluxImageAdapter(BaseImageGenAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backend_module = os.getenv("IMAGE_GEN_OFFLINE_BACKEND")

    @async_tool
    @offline_tool
    def generate_image(self, prompt: str) -> dict[str, Any]:
        """Generate an image from a text prompt using the local FLUX backend.
        Runs as a background job: the immediate response is a job handle;
        poll JobStatusAdapter.check_async_tool(job_id=...) until done, then
        read the generated file paths from the result.
        """
        return self._run(prompt)

    def _generate(self, prompt: str) -> list[bytes]:
        if not self.backend_module:
            raise RuntimeError(
                "No offline image generation backend is installed. Set "
                "IMAGE_GEN_OFFLINE_BACKEND to a module exposing "
                "generate(prompt) -> list[bytes] (e.g. a FLUX.1-schnell shim), "
                "or use the online GeminiImageAdapter."
            )
        backend = importlib.import_module(self.backend_module)
        images = backend.generate(prompt)
        if not images:
            raise RuntimeError(f"Offline backend produced no images for prompt: {prompt!r}")
        if not isinstance(images, list) or not all(isinstance(image, bytes) for image in images):
            raise TypeError("Offline backend must return list[bytes]")
        return images
