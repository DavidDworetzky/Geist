"""
API endpoints for model discovery and listing.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import logging

from agents.architectures.registry import (
    OnlineModelProviders,
    get_models_for_provider,
    get_all_models,
    get_model_by_id,
    get_last_model_sync_time,
    provider_from_string,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelResponse(BaseModel):
    """Response model for a single model."""
    id: str
    name: str
    provider: str
    context_window: Optional[int]
    max_output_tokens: Optional[int]
    supports_vision: bool
    supports_function_calling: bool
    supports_streaming: bool
    #recommended models can be filtered to the top in the UI.
    recommended: bool
    family: Optional[str]


class ModelsListResponse(BaseModel):
    """Response model for list of models grouped by provider."""
    providers: dict[str, List[ModelResponse]]
    last_updated: Optional[datetime]


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
            providers_dict[provider.value] = [
                ModelResponse(**model.to_dict()) for model in models
            ]

        return ModelsListResponse(
            providers=providers_dict,
            last_updated=get_last_model_sync_time()
        )
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/provider/{provider}", response_model=List[ModelResponse])
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
                detail=f"Invalid provider: {provider}. Valid providers: {[p.value for p in OnlineModelProviders]}"
            )

        models = get_models_for_provider(provider_enum)
        return [ModelResponse(**model.to_dict()) for model in models]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


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
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/providers", response_model=List[str])
async def get_providers():
    """
    Get list of available providers.

    Returns:
        List[str]: List of provider names
    """
    return [p.value for p in OnlineModelProviders]
