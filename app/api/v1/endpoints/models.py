"""
API endpoints for model discovery and listing.
"""
import logging
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agents.architectures.llama_server_process import get_llama_server_manager
from agents.architectures.registry import (
    get_all_models,
    get_last_model_sync_time,
    get_model_by_id,
    get_models_for_provider,
    get_provider_ids,
    provider_from_string,
    provider_to_string,
)
from app.services.local_models import get_local_model_manager


logger = logging.getLogger(__name__)

router = APIRouter()


class ModelResponse(BaseModel):
    """Response model for a single model."""
    id: str
    name: str
    provider: str
    context_window: int | None
    max_output_tokens: int | None
    supports_vision: bool
    supports_function_calling: bool
    supports_streaming: bool
    #recommended models can be filtered to the top in the UI.
    recommended: bool
    family: str | None
    backend: str | None = None
    supports_reasoning: bool = False
    gated: bool = False
    requires_remote_code: bool = False
    min_transformers_version: str | None = None
    parameter_count: str | None = None
    activated_parameters: str | None = None
    optional_dependencies: tuple[str, ...] = ()
    local: bool = False
    performance_note: str | None = None
    local_artifacts: list[dict] = Field(default_factory=list)


class ModelsListResponse(BaseModel):
    """Response model for list of models grouped by provider."""
    providers: dict[str, list[ModelResponse]]
    last_updated: datetime | None


def _model_response(model) -> ModelResponse:
    payload = model.to_dict()
    payload["local_artifacts"] = get_local_model_manager().list_artifacts(model.id)
    return ModelResponse(**payload)


@router.get("/", response_model=ModelsListResponse)
async def get_available_models():
    """
    Get all available models grouped by provider.

    Returns:
        ModelsListResponse: All available models grouped by provider
    """
    try:
        all_models = get_all_models()

        providers_dict = {}
        for provider, models in all_models.items():
            providers_dict[provider_to_string(provider)] = [
                _model_response(model) for model in models
            ]

        return ModelsListResponse(
            providers=providers_dict,
            last_updated=get_last_model_sync_time()
        )
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/provider/{provider}", response_model=list[ModelResponse])
async def get_models_by_provider(provider: str):
    """
    Get available models for a specific provider.

    Args:
        provider: Provider name (openai, anthropic, groq, xai, huggingface, offline)

    Returns:
        List[ModelResponse]: Models for the specified provider
    """
    try:
        provider_enum = provider_from_string(provider)
        if provider_enum is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {provider}. Valid providers: {get_provider_ids()}"
            )

        models = get_models_for_provider(provider_enum)
        return [_model_response(model) for model in models]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/model/{model_id:path}", response_model=ModelResponse)
async def get_model(model_id: str):
    """
    Get a specific model by its ID.

    Args:
        model_id: Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")

    Returns:
        ModelResponse: Model details if found
    """
    try:
        model = get_model_by_id(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        return _model_response(model)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/providers", response_model=list[str])
async def get_providers():
    """
    Get list of available providers.

    Returns:
        List[str]: List of provider names
    """
    return get_provider_ids()


@router.get("/local/artifacts")
def get_local_artifacts(model_id: str | None = None):
    """List curated and imported local artifacts with installation progress."""
    return {"artifacts": get_local_model_manager().list_artifacts(model_id)}


@router.get("/local/artifacts/{artifact_id}")
def get_local_artifact(artifact_id: str):
    try:
        return get_local_model_manager().status(artifact_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/local/artifacts/{artifact_id}/download", status_code=202)
def download_local_artifact(artifact_id: str):
    try:
        return get_local_model_manager().request_download(artifact_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/local/artifacts/{artifact_id}/cancel")
def cancel_local_artifact_download(artifact_id: str):
    try:
        return get_local_model_manager().cancel_download(artifact_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.delete("/local/artifacts/{artifact_id}")
def remove_local_artifact(artifact_id: str):
    model_manager = get_local_model_manager()
    runtime_manager = get_llama_server_manager()
    try:
        artifact = model_manager.get_artifact(artifact_id)
        runtime_status = runtime_manager.public_status()
        if (
            runtime_status.get("status") in {"starting", "ready"}
            and runtime_status.get("model_id") == artifact.model_id
        ):
            runtime_manager.stop()
        return model_manager.remove_artifact(artifact_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/local/import")
def import_local_artifact(
    file: UploadFile = File(...),
    model_id: str | None = None,
    display_name: str | None = None,
):
    """Copy an uploaded GGUF into Geist's managed, verified model store."""
    try:
        return get_local_model_manager().import_stream(
            file.file,
            file.filename or "model.gguf",
            model_id=model_id,
            display_name=display_name,
        )
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/local/runtime")
def get_local_runtime_status():
    return get_llama_server_manager().public_status()


@router.post("/local/runtime/stop")
def stop_local_runtime():
    manager = get_llama_server_manager()
    manager.stop()
    return manager.public_status()
