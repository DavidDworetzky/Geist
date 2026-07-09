"""
Agent factory for instantiating LocalAgent and OnlineAgent instances.
"""
from typing import Dict, Any, Optional, Union
import logging
from agents.agent_context import AgentContext

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory class for creating agent instances."""
    
    @staticmethod
    def create_agent(
        agent_type: str,
        agent_context: AgentContext,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        runner_type: Optional[str] = None,
        as_subprocess: bool = False,
        **kwargs
    ):
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
    
    # Model ID prefixes that should use the qwen3 runner
    _QWEN3_PREFIXES = ("qwen/qwen3", "qwen3")

    @staticmethod
    def _infer_runner_type(model: str) -> str:
        """
        Infer the appropriate runner type from a model identifier.

        Args:
            model: Model identifier string

        Returns:
            Runner type string ("qwen3" or "mlx_llama")
        """
        model_lower = model.lower()
        if any(model_lower.startswith(prefix) for prefix in AgentFactory._QWEN3_PREFIXES):
            return "qwen3"
        return "mlx_llama"

    @staticmethod
    def _create_local_agent(
        agent_context: AgentContext,
        model: Optional[str] = None,
        runner_type: Optional[str] = None,
        as_subprocess: bool = False,
        **kwargs
    ):
        """
        Create a LocalAgent instance.

        If runner_type is not specified it is inferred from the model ID.
        For Qwen 3 models a "weights_dir" key in device_config (passed via
        kwargs) tells the runner where to find local safetensors / pretrained
        weights on disk.
        """
        try:
            from agents.local_agent import LocalAgent

            # Default model
            if not model:
                model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

            # Auto-detect runner type from model ID when not explicitly set
            if not runner_type:
                runner_type = AgentFactory._infer_runner_type(model)

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
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        as_subprocess: bool = False,
        **kwargs
    ):
        """Create an OnlineAgent instance."""
        try:
            from agents.online_agent import OnlineAgent
            
            # Default values
            if not endpoint:
                endpoint = "https://api.openai.com/v1/chat/completions"
            if not model:
                model = "gpt-4"
            
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
    def create_from_config(config: Dict[str, Any], agent_context: AgentContext):
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
