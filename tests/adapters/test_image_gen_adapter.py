"""Tests for the image generation adapter."""
import base64

import pytest

from adapters.async_tool import is_async_tool
from adapters.image_gen_adapter import ImageGenAdapter


PNG_BYTES = b"\x89PNG fake image bytes"


@pytest.fixture()
def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_GEN_OUTPUT_DIR", str(tmp_path / "images"))
    return ImageGenAdapter(gemini_api_key="test-key")


def gemini_response(images):
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "here you go"},
                        *[{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(i).decode()}} for i in images],
                    ]
                }
            }
        ]
    }


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_generate_image_is_async_tool(adapter):
    assert is_async_tool(adapter.generate_image)
    assert adapter.enumerate_actions() == ["generate_image"]


def test_online_generation_writes_artifacts(adapter, monkeypatch):
    requests = []

    def fake_post(url, json=None, headers=None, timeout=None):
        requests.append((url, json, headers))
        return FakeResponse(gemini_response([PNG_BYTES, PNG_BYTES]))

    monkeypatch.setattr("adapters.image_gen_adapter.httpx.post", fake_post)

    result = adapter.generate_image(prompt="a cat on a bike", mode="online")

    assert result["mode"] == "online"
    assert result["prompt"] == "a cat on a bike"
    assert len(result["file_paths"]) == 2
    for path in result["file_paths"]:
        with open(path, "rb") as f:
            assert f.read() == PNG_BYTES

    url, payload, headers = requests[0]
    assert "generateContent" in url
    assert payload["contents"][0]["parts"][0]["text"] == "a cat on a bike"
    assert headers["x-goog-api-key"] == "test-key"


def test_online_generation_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("IMAGE_GEN_OUTPUT_DIR", str(tmp_path))
    adapter = ImageGenAdapter(gemini_api_key=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        adapter.generate_image(prompt="a cat", mode="online")


def test_online_generation_no_images_errors(adapter, monkeypatch):
    monkeypatch.setattr(
        "adapters.image_gen_adapter.httpx.post",
        lambda *a, **k: FakeResponse({"candidates": [{"content": {"parts": [{"text": "sorry"}]}}]}),
    )
    with pytest.raises(RuntimeError, match="no image data"):
        adapter.generate_image(prompt="a cat", mode="online")


def test_offline_generation_requires_backend(adapter, monkeypatch):
    monkeypatch.delenv("IMAGE_GEN_OFFLINE_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="offline image generation backend"):
        adapter.generate_image(prompt="a cat", mode="offline")


def test_unknown_mode_errors(adapter):
    with pytest.raises(ValueError, match="Unknown image generation mode"):
        adapter.generate_image(prompt="a cat", mode="sideways")
