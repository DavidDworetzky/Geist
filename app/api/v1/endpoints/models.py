"""
API endpoints for model discovery and listing.
"""
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agents.architectures.registry import (
    get_all_models,
    get_last_model_sync_time,
    get_model_by_id,
    get_models_for_provider,
    get_provider_ids,
    provider_from_string,
    provider_to_string,
)
from app.models.database.geist_user import get_default_user
from app.services.model_downloads import enqueue_model_download, local_model_statuses


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


class ModelsListResponse(BaseModel):
    """Response model for list of models grouped by provider."""
    providers: dict[str, list[ModelResponse]]
    last_updated: datetime | None


class LocalModelStatus(BaseModel):
    """Download state of one catalog model in the packed weights folder."""
    id: str
    name: str
    family: str | None
    backend: str | None
    gated: bool
    parameter_count: str | None
    downloaded: bool
    weights_path: str
    size_bytes: int
    download_status: str | None = None
    download_job_id: int | None = None
    download_error: str | None = None


class DetectedWeightsDirectory(BaseModel):
    """A weights directory present on disk but not mapped to a catalog model."""
    directory: str
    weights_path: str
    size_bytes: int


class LocalModelsResponse(BaseModel):
    """Local weight inventory for the packed weights folder."""
    models: list[LocalModelStatus]
    detected_directories: list[DetectedWeightsDirectory]
    weights_root: str


class ModelDownloadRequest(BaseModel):
    """Request body for queueing a model weight download."""
    model_id: str = Field(min_length=1, max_length=512)
    revision: str | None = Field(default=None, max_length=256)
    force: bool = False


class ModelDownloadResponse(BaseModel):
    """The queued (or already active) download job."""
    job_id: int
    model_id: str
    status: str
    error: str | None = None


def get_current_user():
    """
    Get current user (placeholder - should integrate with actual auth system).
    For now, returns the default user.
    """
    return get_default_user()


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
                ModelResponse(**model.to_dict()) for model in models
            ]

        return ModelsListResponse(
            providers=providers_dict,
            last_updated=get_last_model_sync_time()
        )
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/local", response_model=LocalModelsResponse)
async def get_local_models(current_user = Depends(get_current_user)):
    """
    Get downloadable catalog models with their packed-weights state, plus any
    extra weight directories detected on disk.
    """
    try:
        return local_model_statuses(user_id=current_user.user_id)
    except Exception as e:
        logger.error(f"Error getting local model statuses: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/download", response_model=ModelDownloadResponse, status_code=202)
async def download_model(
    request: ModelDownloadRequest,
    current_user = Depends(get_current_user),
):
    """
    Queue a background download of one local catalog model from Hugging Face
    into the packed weights folder. Poll /local for progress.
    """
    try:
        job = enqueue_model_download(
            request.model_id,
            user_id=current_user.user_id,
            revision=request.revision,
            force=request.force,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error queueing download for '{request.model_id}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    return ModelDownloadResponse(
        job_id=job["job_id"],
        model_id=(job.get("payload") or {}).get("model_id", request.model_id),
        status=job["status"],
        error=job.get("error"),
    )


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
        return [ModelResponse(**model.to_dict()) for model in models]
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

        return ModelResponse(**model.to_dict())
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
