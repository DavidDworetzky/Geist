"""
Image generation adapter, executed asynchronously through the job queue.

generate_image is marked @async_tool: when an agent calls it, the tool
dispatcher enqueues a background job and immediately returns a job handle.
The agent polls JobStatusAdapter.check_async_tool(job_id=...) for the
finished artifact paths, so slow generation never blocks an agent tick or
request thread.

Online mode calls Gemini's image generation API (GEMINI_API_KEY). Offline
mode is an extension point for a local FLUX.1-schnell backend; it reports
clearly when no local backend is installed rather than dragging heavy
diffusion dependencies into the default install.
"""
import base64
import datetime
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

from adapters.async_tool import async_tool
from adapters.base_adapter import BaseAdapter


logger = logging.getLogger(__name__)

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"
DEFAULT_OUTPUT_DIR = "output/images"


class ImageGenAdapter(BaseAdapter):

    def __init__(self, gemini_api_key: str | None = None, **kwargs):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_base_url = os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL).rstrip("/")
        self.gemini_image_model = os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL)
        self.output_dir = os.getenv("IMAGE_GEN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
        self.timeout = float(os.getenv("IMAGE_GEN_TIMEOUT_SECONDS", "120"))

    def enumerate_actions(self):
        return ["generate_image"]

    @async_tool
    def generate_image(self, prompt: str, mode: str = "online") -> dict[str, Any]:
        """Generate an image from a text prompt. Runs as a background job:
        the immediate response is a job handle; poll
        JobStatusAdapter.check_async_tool(job_id=...) until done, then read
        the generated file paths from the result.

        mode is "online" (Gemini image generation API) or "offline"
        (local FLUX backend, if installed).
        """
        if mode == "online":
            file_paths = self._generate_online(prompt)
        elif mode == "offline":
            file_paths = self._generate_offline(prompt)
        else:
            raise ValueError(f"Unknown image generation mode '{mode}'; use 'online' or 'offline'")
        return {"prompt": prompt, "mode": mode, "file_paths": file_paths}

    def _generate_online(self, prompt: str) -> list[str]:
        """Generate via Gemini and write the returned images to disk."""
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured for online image generation")

        url = f"{self.gemini_base_url}/models/{self.gemini_image_model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        response = httpx.post(
            url,
            json=payload,
            headers={"x-goog-api-key": self.gemini_api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        images = []
        for candidate in data.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    images.append(base64.b64decode(inline["data"]))
        if not images:
            raise RuntimeError(f"Gemini returned no image data for prompt: {prompt!r}")
        return self._write_artifacts(images)

    def _generate_offline(self, prompt: str) -> list[str]:
        """
        Local generation extension point. A FLUX.1-schnell (or other local
        diffusion) backend can be plugged in by providing a module exposing
        generate(prompt) -> list[bytes], named via IMAGE_GEN_OFFLINE_BACKEND.
        """
        backend_module = os.getenv("IMAGE_GEN_OFFLINE_BACKEND")
        if not backend_module:
            raise RuntimeError(
                "No offline image generation backend is installed. Set "
                "IMAGE_GEN_OFFLINE_BACKEND to a module exposing "
                "generate(prompt) -> list[bytes], or use mode='online'."
            )
        import importlib

        backend = importlib.import_module(backend_module)
        images = backend.generate(prompt)
        if not images:
            raise RuntimeError(f"Offline backend produced no images for prompt: {prompt!r}")
        return self._write_artifacts(images)

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
