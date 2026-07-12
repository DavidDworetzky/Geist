"""Tests for the Gemini (online) and FLUX (offline) image generation adapters."""
import base64
import sys
import types

import pytest

from adapters.adapter_registry import find_adapter_classes
from adapters.async_tool import is_async_tool
from adapters.flux_image_adapter import FluxImageAdapter
from adapters.gemini_image_adapter import GeminiImageAdapter
from adapters.tool_modes import OFFLINE, ONLINE, tool_mode


PNG_BYTES = b"\x89PNG fake image bytes"


@pytest.fixture()
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_GEN_OUTPUT_DIR", str(tmp_path / "images"))
    return tmp_path / "images"


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


def test_mode_decorators_and_async_markers():
    assert tool_mode(GeminiImageAdapter.generate_image) == ONLINE
    assert tool_mode(FluxImageAdapter.generate_image) == OFFLINE
    assert is_async_tool(GeminiImageAdapter.generate_image)
    assert is_async_tool(FluxImageAdapter.generate_image)


def test_both_adapters_discovered_but_not_the_abstract_base():
    names = [name for name, _ in find_adapter_classes()]
    assert "GeminiImageAdapter" in names
    assert "FluxImageAdapter" in names
    assert "BaseImageGenAdapter" not in names


def test_gemini_generation_writes_artifacts(output_dir, monkeypatch):
    requests = []

    def fake_post(url, json=None, headers=None, timeout=None):
        requests.append((url, json, headers))
        return FakeResponse(gemini_response([PNG_BYTES, PNG_BYTES]))

    monkeypatch.setattr("adapters.gemini_image_adapter.httpx.post", fake_post)
    adapter = GeminiImageAdapter(gemini_api_key="test-key")

    result = adapter.generate_image(prompt="a cat on a bike")

    assert result["mode"] == ONLINE
    assert result["adapter"] == "GeminiImageAdapter"
    assert result["prompt"] == "a cat on a bike"
    assert len(result["file_paths"]) == 2
    for path in result["file_paths"]:
        with open(path, "rb") as f:
            assert f.read() == PNG_BYTES

    url, payload, headers = requests[0]
    assert "generateContent" in url
    assert payload["contents"][0]["parts"][0]["text"] == "a cat on a bike"
    assert headers["x-goog-api-key"] == "test-key"


def test_gemini_requires_api_key(output_dir, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adapter = GeminiImageAdapter(gemini_api_key=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        adapter.generate_image(prompt="a cat")


def test_gemini_no_images_errors(output_dir, monkeypatch):
    monkeypatch.setattr(
        "adapters.gemini_image_adapter.httpx.post",
        lambda *a, **k: FakeResponse({"candidates": [{"content": {"parts": [{"text": "sorry"}]}}]}),
    )
    adapter = GeminiImageAdapter(gemini_api_key="test-key")
    with pytest.raises(RuntimeError, match="no image data"):
        adapter.generate_image(prompt="a cat")


def test_flux_requires_backend(output_dir, monkeypatch):
    monkeypatch.delenv("IMAGE_GEN_OFFLINE_BACKEND", raising=False)
    adapter = FluxImageAdapter()
    with pytest.raises(RuntimeError, match="offline image generation backend"):
        adapter.generate_image(prompt="a cat")


def test_flux_uses_configured_backend(output_dir, monkeypatch):
    backend = types.ModuleType("fake_flux_backend")
    backend.generate = lambda prompt: [PNG_BYTES]
    monkeypatch.setitem(sys.modules, "fake_flux_backend", backend)
    monkeypatch.setenv("IMAGE_GEN_OFFLINE_BACKEND", "fake_flux_backend")

    adapter = FluxImageAdapter()
    result = adapter.generate_image(prompt="a dog")

    assert result["mode"] == OFFLINE
    assert result["adapter"] == "FluxImageAdapter"
    assert len(result["file_paths"]) == 1
    with open(result["file_paths"][0], "rb") as f:
        assert f.read() == PNG_BYTES


def test_flux_empty_backend_output_errors(output_dir, monkeypatch):
    backend = types.ModuleType("empty_flux_backend")
    backend.generate = lambda prompt: []
    monkeypatch.setitem(sys.modules, "empty_flux_backend", backend)
    monkeypatch.setenv("IMAGE_GEN_OFFLINE_BACKEND", "empty_flux_backend")

    adapter = FluxImageAdapter()
    with pytest.raises(RuntimeError, match="no images"):
        adapter.generate_image(prompt="a dog")
