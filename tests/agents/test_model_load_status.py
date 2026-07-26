import asyncio

from agents.model_load_status import ModelLoadStatusRegistry, model_load_status_registry
from app.api.v1.endpoints.models import get_model_load_status


def test_model_load_registry_tracks_lifecycle() -> None:
    registry = ModelLoadStatusRegistry()

    unloaded = registry.get("org/model")
    loading = registry.mark_loading("org/model", "Loading model files.")
    ready = registry.mark_ready("org/model")

    assert unloaded.state == "unloaded"
    assert unloaded.started_at is None
    assert loading.state == "loading"
    assert loading.started_at is not None
    assert ready.state == "ready"
    assert ready.started_at == loading.started_at


def test_model_load_registry_records_failure() -> None:
    registry = ModelLoadStatusRegistry()
    registry.mark_loading("org/model", "Loading model files.")

    failed = registry.mark_failed("org/model", "Model failed to load.")

    assert failed.state == "failed"
    assert failed.detail == "Model failed to load."


def test_model_status_endpoint_reports_process_local_state() -> None:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_load_status_registry.clear()
    try:
        unloaded = asyncio.run(get_model_load_status(model_id))
        model_load_status_registry.mark_loading(model_id, "Loading cached weights.")
        loading = asyncio.run(get_model_load_status(model_id))

        assert unloaded.state == "unloaded"
        assert loading.state == "loading"
        assert loading.detail == "Loading cached weights."
    finally:
        model_load_status_registry.clear()


def test_remote_model_status_is_always_ready() -> None:
    status = asyncio.run(get_model_load_status("gpt-4"))

    assert status.state == "ready"
    assert status.started_at is None
