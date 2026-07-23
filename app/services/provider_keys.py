"""
Provider API key management and resolution.

Stored credentials (entered through the Settings UI) take precedence over
environment variables so a key can be rotated without restarting the server.
Environment variables remain the fallback so existing .env-based deployments
keep working unchanged.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from agents.model_catalog import PROVIDERS
from app.models.database.provider_credential import (
    delete_provider_credential,
    get_provider_credential,
    list_provider_credentials,
    upsert_provider_credential,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManagedProvider:
    """One provider whose API key can be entered in the UI."""

    id: str
    name: str
    api_key_env: str
    description: str
    supports_base_url: bool = False
    base_url_env: str | None = None


def _managed_providers() -> dict[str, ManagedProvider]:
    providers: dict[str, ManagedProvider] = {}
    for spec in PROVIDERS.values():
        providers[spec.id] = ManagedProvider(
            id=spec.id,
            name=spec.name,
            api_key_env=spec.api_key_env,
            description=f"API key for {spec.name} chat completions.",
            supports_base_url=spec.base_url_env is not None,
            base_url_env=spec.base_url_env,
        )
    # Providers used by the app but not represented in the catalog's
    # OpenAI-compatible PROVIDERS map.
    providers["anthropic"] = ManagedProvider(
        id="anthropic",
        name="Anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        description="API key for Anthropic chat completions.",
    )
    providers["huggingface"] = ManagedProvider(
        id="huggingface",
        name="Hugging Face",
        api_key_env="HUGGING_FACE_HUB_TOKEN",
        description=(
            "Access token used to download model weights; required for gated "
            "repositories such as Llama and Gemma."
        ),
    )
    return providers


MANAGED_PROVIDERS: dict[str, ManagedProvider] = _managed_providers()


def is_managed_provider(provider_id: str) -> bool:
    return provider_id.strip().lower() in MANAGED_PROVIDERS


def _mask_key(api_key: str) -> str:
    """Return a display hint that never reveals the key body."""
    tail = api_key[-4:] if len(api_key) >= 8 else ""
    return f"****{tail}"


def resolve_api_key(provider_id: str, user_id: int | None = None) -> str | None:
    """
    Resolve the API key for a provider: stored credential first, env second.

    Falls back cleanly when the database is unavailable (e.g. unit tests that
    construct agents without configuring a database).
    """
    provider = MANAGED_PROVIDERS.get(provider_id.strip().lower())
    if provider is None:
        return None

    if user_id is None:
        user_id = _default_user_id()
    if user_id is not None:
        try:
            credential = get_provider_credential(user_id, provider.id)
        except Exception:
            logger.debug(
                "Stored credential lookup failed for provider '%s'; using environment",
                provider.id,
                exc_info=True,
            )
            credential = None
        if credential and credential.api_key:
            return credential.api_key

    env_value = os.getenv(provider.api_key_env)
    if env_value:
        return env_value
    if provider.id == "huggingface":
        return os.getenv("HF_TOKEN")
    return None


def resolve_base_url(provider_id: str, user_id: int | None = None) -> str | None:
    """Resolve a stored or environment-configured base URL override."""
    provider = MANAGED_PROVIDERS.get(provider_id.strip().lower())
    if provider is None or not provider.supports_base_url:
        return None

    if user_id is None:
        user_id = _default_user_id()
    if user_id is not None:
        try:
            credential = get_provider_credential(user_id, provider.id)
        except Exception:
            credential = None
        if credential and credential.base_url:
            return credential.base_url.rstrip("/")

    if provider.base_url_env:
        configured = os.getenv(provider.base_url_env)
        if configured:
            return configured.rstrip("/")
    return None


def resolve_huggingface_token(user_id: int | None = None) -> str | None:
    """Convenience wrapper used by model downloads and gated weight loading."""
    return resolve_api_key("huggingface", user_id=user_id)


def provider_key_statuses(user_id: int) -> list[dict]:
    """
    Describe every managed provider's key state for the Settings UI.

    Raw keys are never included: stored keys surface only as a masked hint.
    """
    stored = {}
    try:
        stored = {credential.provider_id: credential for credential in
                  list_provider_credentials(user_id)}
    except Exception:
        logger.warning("Could not list stored provider credentials", exc_info=True)

    statuses = []
    for provider in MANAGED_PROVIDERS.values():
        credential = stored.get(provider.id)
        env_configured = bool(os.getenv(provider.api_key_env)) or (
            provider.id == "huggingface" and bool(os.getenv("HF_TOKEN"))
        )
        statuses.append({
            "id": provider.id,
            "name": provider.name,
            "description": provider.description,
            "api_key_env": provider.api_key_env,
            "env_configured": env_configured,
            "has_stored_key": credential is not None,
            "key_hint": _mask_key(credential.api_key) if credential else None,
            "supports_base_url": provider.supports_base_url,
            "base_url": credential.base_url if credential else None,
            "updated_at": credential.update_date.isoformat() if credential else None,
        })
    return statuses


def store_provider_key(
    user_id: int,
    provider_id: str,
    api_key: str,
    base_url: str | None = None,
) -> dict:
    """Store one provider key and return its masked status entry."""
    normalized = provider_id.strip().lower()
    provider = MANAGED_PROVIDERS.get(normalized)
    if provider is None:
        raise ValueError(f"Unknown provider: {provider_id}")
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("API key must not be empty")
    if base_url is not None:
        base_url = base_url.strip() or None
    if base_url and not provider.supports_base_url:
        raise ValueError(f"Provider '{provider.id}' does not accept a base URL")

    credential = upsert_provider_credential(user_id, provider.id, api_key, base_url=base_url)
    return {
        "id": provider.id,
        "name": provider.name,
        "has_stored_key": True,
        "key_hint": _mask_key(credential.api_key),
        "base_url": credential.base_url,
    }


def remove_provider_key(user_id: int, provider_id: str) -> bool:
    """Delete a stored provider key; returns False when none existed."""
    normalized = provider_id.strip().lower()
    if normalized not in MANAGED_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_id}")
    return delete_provider_credential(user_id, normalized)


def _default_user_id() -> int | None:
    """Resolve the default user without failing when the DB is unavailable."""
    try:
        from app.models.database.geist_user import get_default_user

        return get_default_user().user_id
    except Exception:
        return None
