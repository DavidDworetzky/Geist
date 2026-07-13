"""
Online image generation via Gemini's image generation API.

generate_image is an @async_tool: the agent gets a job handle immediately
and polls JobStatusAdapter.check_async_tool(job_id=...) for the artifact
paths. Configure with GEMINI_API_KEY (also injected through
app/environment.py), GEMINI_BASE_URL, and GEMINI_IMAGE_MODEL.
"""
import base64
import logging
import os
from typing import Any

import httpx

from adapters.async_tool import async_tool
from adapters.image_gen_base import BaseImageGenAdapter
from adapters.tool_modes import online_tool


logger = logging.getLogger(__name__)

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"


class GeminiImageAdapter(BaseImageGenAdapter):

    def __init__(self, gemini_api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_base_url = os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL).rstrip("/")
        self.gemini_image_model = os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL)

    @async_tool
    @online_tool
    def generate_image(self, prompt: str) -> dict[str, Any]:
        """Generate an image from a text prompt using the hosted Gemini API.
        Runs as a background job: the immediate response is a job handle;
        poll JobStatusAdapter.check_async_tool(job_id=...) until done, then
        read the generated file paths from the result.
        """
        return self._run(prompt)

    def _generate(self, prompt: str) -> list[bytes]:
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
        return images
