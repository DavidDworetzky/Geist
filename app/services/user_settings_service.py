"""
Service layer for user settings management.
"""

import logging
from typing import Any, Literal

from agents.agent_context import AgentContext
from agents.base_agent import BaseAgent
from agents.factory import AgentFactory
from agents.model_ids import canonicalize_local_model_id
from app.models.database.geist_user import get_default_workspace
from app.models.database.user_settings import (
    UserSettingsModel,
    get_or_create_user_settings,
    get_user_settings,
    update_detected_llama_backend_if_unset,
    update_user_settings,
)
from app.models.user_settings import (
    AgentConfigRequest,
    AgentFactoryConfig,
    UserSettingsResponse,
    UserSettingsUpdate,
)


logger = logging.getLogger(__name__)


def _to_user_settings_response(settings_model: UserSettingsModel) -> UserSettingsResponse:
    response = UserSettingsResponse.model_validate(settings_model)
    return response.model_copy(
        update={"default_local_model": canonicalize_local_model_id(response.default_local_model)}
    )


class UserSettingsService:
    """Service for managing workspace settings and agent configuration."""

    @staticmethod
    def get_workspace_settings_by_id(workspace_id: int) -> UserSettingsResponse | None:
        """
        Get settings owned by a workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            UserSettingsResponse if found, None otherwise
        """
        settings_model = get_user_settings(workspace_id)
        if settings_model:
            return _to_user_settings_response(settings_model)
        return None

    @staticmethod
    def get_or_create_workspace_settings_by_id(workspace_id: int) -> UserSettingsResponse:
        """
        Get workspace settings, creating defaults when they do not exist.

        Args:
            workspace_id: Workspace ID

        Returns:
            UserSettingsResponse
        """
        settings_model = get_or_create_user_settings(workspace_id)
        return _to_user_settings_response(settings_model)

    @staticmethod
    def update_workspace_settings_by_id(
        workspace_id: int,
        updates: UserSettingsUpdate,
        *,
        allow_llama_redetection: bool = False,
    ) -> UserSettingsResponse | None:
        """
        Update user settings.

        Args:
            workspace_id: Workspace ID
            updates: Settings updates

        Returns:
            Updated UserSettingsResponse if successful, None if user not found
        """
        # Preserve explicitly supplied nulls so callers can clear a previously
        # selected concrete artifact when switching back to a catalog model.
        update_dict = updates.model_dump(exclude_unset=True)
        current_settings = get_user_settings(workspace_id)
        if current_settings is None:
            return None

        if "default_local_model" in update_dict:
            update_dict["default_local_model"] = canonicalize_local_model_id(
                update_dict["default_local_model"]
            )

        current_local_model = canonicalize_local_model_id(current_settings.default_local_model)

        compute_keys = {"llama_backend", "llama_gpu_device_ids"}
        compute_supplied = any(key in update_dict for key in compute_keys)
        compute_changed = any(
            key in update_dict and update_dict[key] != getattr(current_settings, key)
            for key in compute_keys
        )
        next_backend = update_dict.get("llama_backend", current_settings.llama_backend)
        next_device_ids = update_dict.get(
            "llama_gpu_device_ids", current_settings.llama_gpu_device_ids
        )
        if compute_changed or (compute_supplied and next_backend == "gpu"):
            if (
                next_backend is None
                and current_settings.llama_backend is not None
                and not allow_llama_redetection
            ):
                raise ValueError(
                    "llama.cpp backend detection can only be reset with Reset to Defaults"
                )
            if next_backend is None:
                if next_device_ids and not allow_llama_redetection:
                    raise ValueError("Pending llama.cpp detection cannot select GPU devices")
                update_dict["llama_gpu_device_ids"] = []

            resetting_detection = allow_llama_redetection and next_backend is None
            if not resetting_detection:
                from agents.architectures.llama_devices import get_llama_device_service

                try:
                    inventory = get_llama_device_service().inventory()
                except RuntimeError as error:
                    raise ValueError(
                        "GPU discovery failed; retry saving compute settings after a short wait"
                    ) from error
                if inventory.managed_by_environment:
                    raise ValueError("llama.cpp compute selection is managed by the environment")
                if next_backend == "gpu":
                    if not next_device_ids:
                        raise ValueError("Select at least one llama.cpp GPU device")
                    if len(set(next_device_ids)) != len(next_device_ids):
                        raise ValueError("llama.cpp GPU device selections must be unique")
                    if not inventory.available:
                        raise ValueError("Managed llama.cpp GPU selection is unavailable")
                    # Validate against the exact snapshot used for the rest of this
                    # update. Its resolver also accepts unambiguous compatibility
                    # IDs from settings written before stable hardware IDs existed.
                    canonical_device_ids = inventory.resolve_device_ids(next_device_ids)
                    update_dict["llama_gpu_device_ids"] = list(canonical_device_ids)
                elif next_backend == "cpu":
                    update_dict["llama_gpu_device_ids"] = []
        if (
            "default_local_model" in update_dict
            and "default_local_artifact_id" not in update_dict
            and update_dict["default_local_model"] != current_local_model
        ):
            update_dict["default_local_artifact_id"] = None

        selected_artifact_id = update_dict.get("default_local_artifact_id")
        if selected_artifact_id:
            from app.services.local_models import get_local_model_manager

            selected_model = update_dict.get("default_local_model") or current_local_model
            manager = get_local_model_manager()
            try:
                artifact = manager.get_artifact(str(selected_artifact_id))
            except KeyError as error:
                raise ValueError(str(error)) from error
            if artifact.model_id != selected_model:
                raise ValueError(
                    f"Artifact {artifact.id} belongs to {artifact.model_id}, not {selected_model}"
                )
            artifact_status = manager.status(artifact.id)
            if artifact_status.get("supported") is False:
                raise ValueError(f"Artifact {artifact.id} is unavailable on this platform")

        # Backend validation: auto-infer agent_type based on model/provider changes
        # This acts as a safety net if the frontend doesn't set agent_type correctly
        if "default_agent_type" not in update_dict or update_dict.get("default_agent_type") is None:
            # If online model or online provider is being set, infer agent_type as 'online'
            if "default_online_model" in update_dict or "default_online_provider" in update_dict:
                provider = update_dict.get("default_online_provider", "")
                # Only set to online if provider is not 'offline'
                if provider != "offline":
                    update_dict["default_agent_type"] = "online"
                    logger.info(
                        "Auto-inferred agent_type='online' based on online model/provider update"
                    )
            # If local model is being set, infer agent_type as 'local'
            elif "default_local_model" in update_dict or selected_artifact_id:
                update_dict["default_agent_type"] = "local"
                logger.info("Auto-inferred agent_type='local' based on local model update")

        settings_model = update_user_settings(workspace_id, update_dict)
        if settings_model:
            return _to_user_settings_response(settings_model)
        return None

    @staticmethod
    def persist_detected_llama_backend(
        user_id: int,
        backend: Literal["cpu", "gpu"],
        device_ids: tuple[str, ...],
    ) -> UserSettingsResponse | None:
        """Persist a first-use result without overwriting a concurrent user choice."""

        current_settings = get_user_settings(user_id)
        if current_settings is None or current_settings.llama_backend is not None:
            return UserSettingsService.get_workspace_settings_by_id(user_id)
        if backend not in {"cpu", "gpu"}:
            raise ValueError("Detected llama.cpp backend must be cpu or gpu")
        update_detected_llama_backend_if_unset(
            user_id,
            backend,
            list(device_ids) if backend == "gpu" else [],
        )
        return UserSettingsService.get_workspace_settings_by_id(user_id)

    @staticmethod
    def get_default_workspace_settings() -> UserSettingsResponse:
        """Return settings for the singleton local workspace."""
        workspace = get_default_workspace()
        return UserSettingsService.get_or_create_workspace_settings_by_id(workspace.workspace_id)

    @staticmethod
    def create_agent_from_workspace_settings(
        workspace_id: int, agent_context: AgentContext, overrides: AgentConfigRequest | None = None
    ) -> BaseAgent:
        """
        Create an agent instance based on workspace settings and optional overrides.

        Args:
            workspace_id: Workspace ID to get settings for
            agent_context: Agent context object
            overrides: Optional configuration overrides

        Returns:
            Agent instance
        """
        settings = UserSettingsService.get_or_create_workspace_settings_by_id(workspace_id)

        # Create agent factory config
        factory_config = AgentFactoryConfig.from_user_settings(settings, overrides)

        logger.info(f"Creating agent with config: {factory_config}")

        factory_kwargs: dict[str, Any] = {
            "agent_type": factory_config.agent_type,
            "agent_context": agent_context,
            "model": factory_config.model,
            "endpoint": factory_config.endpoint,
            "api_key": factory_config.api_key,
            "runner_type": factory_config.runner_type,
            "device_config": factory_config.device_config,
            "generation_config": factory_config.generation_config,
        }
        if factory_config.agent_type == "online":
            factory_kwargs["backup_providers"] = [
                provider.model_dump() for provider in factory_config.backup_providers
            ]

        agent = AgentFactory.create_agent(
            **factory_kwargs,
        )

        return agent

    @staticmethod
    def create_agent_from_default_workspace(
        agent_context: AgentContext, overrides: AgentConfigRequest | None = None
    ) -> BaseAgent:
        """
        Create an agent instance for the default workspace.

        Args:
            agent_context: Agent context object
            overrides: Optional configuration overrides

        Returns:
            Agent instance
        """
        workspace = get_default_workspace()
        return UserSettingsService.create_agent_from_workspace_settings(
            workspace.workspace_id, agent_context, overrides
        )
