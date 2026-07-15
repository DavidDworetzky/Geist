import os
import uuid
from typing import Any

import httpx

from adapters.base_adapter import BaseAdapter
from agents.models.chat_result import WorkArtifact


class ImageGenerationAdapter(BaseAdapter):
    """Adapter for creating image artifacts through OpenAI-compatible image APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_image_bytes: int = 25_000_000,
        **kwargs,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.getenv("OPENAI_IMAGE_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.getenv("OPENAI_IMAGE_MODEL") or "gpt-image-1"
        self.timeout = timeout
        self.max_image_bytes = max_image_bytes

    def enumerate_actions(self) -> list[str]:
        return ["generate_image"]

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str | None = None,
        style: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for image generation")
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required for image generation")

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt.strip(),
            "size": size,
        }
        if quality:
            payload["quality"] = quality
        if style:
            payload["style"] = style

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/images/generations",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

        if response.status_code >= 400:
            raise RuntimeError(f"Image generation failed: {response.status_code} {response.text}")

        data = response.json()
        images = data.get("data") or []
        if not images:
            raise RuntimeError("Image generation response did not include image data")

        image = images[0]
        filename = f"generated-image-{uuid.uuid4().hex[:8]}.png"
        if image.get("b64_json"):
            import base64

            image_bytes = base64.b64decode(image["b64_json"], validate=True)
            if len(image_bytes) > self.max_image_bytes:
                raise RuntimeError("Generated image exceeded the configured size limit")
            artifact = WorkArtifact.from_bytes(
                image_bytes,
                kind="image",
                mime_type="image/png",
                filename=filename,
            )
        elif image.get("url"):
            artifact = WorkArtifact.from_url(
                image["url"],
                kind="image",
                mime_type="image/png",
                filename=filename,
            )
        else:
            raise RuntimeError("Image generation response did not include b64_json or url")

        return {
            "summary": f"Generated image for: {prompt.strip()}",
            "artifact": artifact,
        }
