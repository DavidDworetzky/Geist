"""
Agent factory for instantiating LocalAgent and OnlineAgent instances.
"""
import logging
import os
import platform
import sys
from typing import Any

from agents.agent_context import AgentContext
from agents.base_agent import BaseAgent


logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory class for creating agent instances."""

    @staticmethod
    def create_agent(
        agent_type: str,
        agent_context: AgentContext,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        runner_type: str | None = None,
        as_subprocess: bool = False,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent instance based on the specified type and configuration.

        Args:
            agent_type: Type of agent to create ('local' or 'online')
            agent_context: Agent context object
            model: Model identifier
            endpoint: API endpoint (for online agents)
            api_key: API key (for online agents)
            runner_type: Type of runner for local agents
            as_subprocess: Whether to run as subprocess
            **kwargs: Additional arguments for agent initialization

        Returns:
            Agent instance
        """
        agent_type = agent_type.lower()

        if agent_type == "local":
            return AgentFactory._create_local_agent(
                agent_context=agent_context,
                model=model,
                runner_type=runner_type,
                as_subprocess=as_subprocess,
                **kwargs
            )
        elif agent_type == "online":
            return AgentFactory._create_online_agent(
                agent_context=agent_context,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                as_subprocess=as_subprocess,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}. Must be 'local' or 'online'")
    @staticmethod
    def _infer_runner_type(model: str) -> str:
        """Infer a backend from catalog capabilities, not provider branding."""
        from agents.model_catalog import infer_model_spec

        spec = infer_model_spec(model)
        if spec:
            if not spec.local:
                raise ValueError(
                    f"{model} is server-backed. Create an online agent with the "
                    f"'{spec.provider}' provider or an OpenAI-compatible endpoint."
                )
            if sys.platform in {"win32", "linux"}:
                return "llama_server"
            if sys.platform == "darwin" and platform.machine().lower() in {
                "arm64",
                "aarch64",
            }:
                return "mlx_llama"
            return spec.backend

        # Preserve legacy bare Llama weight-directory names. Unknown Hugging
        # Face causal models intentionally fall back to the generic runner so
        # adding a compatible model does not require a code change.
        if sys.platform in {"win32", "linux"}:
            return "llama_server"
        if model.lower().startswith(("meta-llama-", "llama_3", "llama-3")):
            return "mlx_llama"
        return "transformers"

    @staticmethod
    def _create_local_agent(
        agent_context: AgentContext,
        model: str | None = None,
        runner_type: str | None = None,
        as_subprocess: bool = False,
        **kwargs
    ) -> BaseAgent:
        """
        Create a LocalAgent instance.

        If runner_type is not specified it is inferred from model catalog
        metadata. A ``weights_dir`` key can point any runner at local weights.
        """
        try:
            from agents.local_agent import LocalAgent

            # Default model
            if not model:
                model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

            configured_runner = (os.getenv("GEIST_LOCAL_RUNNER") or "").strip()
            runner_was_explicit = runner_type is not None or bool(configured_runner)
            # Auto-detect runner type from model ID when not explicitly set
            if not runner_type:
                runner_type = configured_runner or AgentFactory._infer_runner_type(model)

            if not runner_was_explicit and runner_type == "mlx_llama":
                from agents.model_catalog import infer_model_spec

                spec = infer_model_spec(model)
                if spec is not None and spec.family != "llama":
                    device_config = kwargs.pop("device_config", None) or {}
                    device_config.setdefault("implementation", "mlx_lm")
                    kwargs["device_config"] = device_config

            # Automatic selection protects users from accidental 32B+ loads.
            # A deliberate Transformers override is the opt-in for capable
            # hardware and is forwarded as a narrowly scoped runner flag.
            if runner_was_explicit and runner_type == "transformers":
                from agents.model_catalog import infer_model_spec

                spec = infer_model_spec(model)
                if spec and not spec.local:
                    device_config = kwargs.pop("device_config", None) or {}
                    device_config.setdefault("allow_server_backed", True)
                    kwargs["device_config"] = device_config

            # Propagate weights_dir into device_config so the runner can
            # load safetensors / pretrained weights from a custom path.
            if "weights_dir" in kwargs:
                device_config = kwargs.pop("device_config", None) or {}
                device_config["weights_dir"] = kwargs.pop("weights_dir")
                kwargs["device_config"] = device_config

            logger.info(f"Creating LocalAgent with runner: {runner_type}, model: {model}")
            return LocalAgent(
                agent_context=agent_context,
                model_id=model,
                runner_type=runner_type,
                as_subprocess=as_subprocess,
                **kwargs
            )
        except ImportError as e:
            logger.error(f"Failed to import LocalAgent: {e}")
            raise

    @staticmethod
    def _create_online_agent(
        agent_context: AgentContext,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        as_subprocess: bool = False,
        **kwargs
    ) -> BaseAgent:
        """Create an OnlineAgent instance."""
        try:
            from agents.online_agent import OnlineAgent

            if not model:
                model = "gpt-4"
            if not endpoint:
                from agents.model_catalog import get_provider_endpoint, infer_model_spec

                spec = infer_model_spec(model)
                if spec and spec.backend == "openai_compatible":
                    endpoint = get_provider_endpoint(spec.provider)
                    if not endpoint:
                        raise ValueError(
                            f"{model} requires an OpenAI-compatible endpoint. Pass endpoint=... "
                            "or set OPENAI_COMPATIBLE_BASE_URL."
                        )
                else:
                    endpoint = "https://api.openai.com/v1"

            logger.info(f"Creating OnlineAgent with endpoint: {endpoint}, model: {model}")
            return OnlineAgent(
                agent_context=agent_context,
                base_url=endpoint,
                model=model,
                api_key=api_key,
                as_subprocess=as_subprocess,
                **kwargs
            )
        except ImportError as e:
            logger.error(f"Failed to import OnlineAgent: {e}")
            raise

    @staticmethod
    def create_from_config(config: dict[str, Any], agent_context: AgentContext) -> BaseAgent:
        """
        Create an agent instance from a configuration dictionary.

        Args:
            config: Configuration dictionary containing agent parameters
            agent_context: Agent context object

        Returns:
            Agent instance
        """
        required_fields = ["agent_type"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        return AgentFactory.create_agent(
            agent_type=config["agent_type"],
            agent_context=agent_context,
            model=config.get("model"),
            endpoint=config.get("endpoint"),
            api_key=config.get("api_key"),
            runner_type=config.get("runner_type"),
            as_subprocess=config.get("as_subprocess", False),
            **{k: v for k, v in config.items() if k not in [
                "agent_type", "model", "endpoint", "api_key", "runner_type", "as_subprocess"
            ]}
        )
