from unittest.mock import patch

from agents.agent_type import AgentType
from app import main as geist_main
from app.models.user_settings import AgentFactoryConfig


class RecordingLocalAgent:
    def __init__(self, artifact_id: str, events: list[tuple[str, str]]):
        self.artifact_id = artifact_id
        self.events = events

    def phase_out(self) -> None:
        self.events.append(("stop", self.artifact_id))


def _factory_config(
    artifact_id: str,
    *,
    model: str = "Qwen/Qwen3-4B-GGUF",
    runner_type: str | None = "llama_server",
    temperature: float = 0.7,
) -> AgentFactoryConfig:
    return AgentFactoryConfig(
        agent_type="local",
        model=model,
        runner_type=runner_type,
        device_config={"artifact_id": artifact_id},
        generation_config={"temperature": temperature, "max_tokens": 512},
    )


def test_local_artifact_change_phases_out_old_agent_before_starting_new_server() -> None:
    saved_cache = {
        agent_type: geist_main.agent_cache[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    saved_signatures = {
        agent_type: geist_main._agent_cache_signatures[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    events: list[tuple[str, str]] = []
    selected = {"config": _factory_config("artifact-a")}

    def create_agent(config: AgentFactoryConfig) -> RecordingLocalAgent:
        artifact_id = config.device_config["artifact_id"]
        events.append(("start", artifact_id))
        return RecordingLocalAgent(artifact_id, events)

    try:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()

        with (
            patch(
                "app.main._get_local_agent_factory_config",
                side_effect=lambda: selected["config"],
            ),
            patch("app.main._create_local_agent", side_effect=create_agent) as create,
        ):
            first = geist_main.get_active_agent(AgentType.LLAMA)
            assert geist_main.get_active_agent(AgentType.LOCALAGENT) is first
            assert create.call_count == 1
            assert geist_main.agent_cache[AgentType.LLAMA] is first
            assert geist_main.agent_cache[AgentType.LOCALAGENT] is first

            selected["config"] = _factory_config("artifact-b")
            second = geist_main.get_active_agent(AgentType.LOCALAGENT)

        assert second is not first
        assert events == [
            ("start", "artifact-a"),
            ("stop", "artifact-a"),
            ("start", "artifact-b"),
        ]
        assert geist_main.agent_cache[AgentType.LLAMA] is second
        assert geist_main.agent_cache[AgentType.LOCALAGENT] is second
    finally:
        with geist_main._agent_cache_lock:
            test_agents = geist_main._clear_local_agent_cache()
            for test_agent in test_agents:
                geist_main._phase_out_agent_safely(test_agent)
            for agent_type in geist_main._LOCAL_AGENT_TYPES:
                geist_main.agent_cache[agent_type] = saved_cache[agent_type]
                geist_main._agent_cache_signatures[agent_type] = saved_signatures[agent_type]


def test_local_agent_signature_covers_model_runner_artifact_and_generation() -> None:
    base = _factory_config("artifact-a")
    signature = geist_main._local_agent_configuration_signature(base)

    assert signature != geist_main._local_agent_configuration_signature(
        _factory_config("artifact-a", model="Qwen/Qwen3-8B-GGUF")
    )
    assert signature != geist_main._local_agent_configuration_signature(
        _factory_config("artifact-a", runner_type="transformers")
    )
    assert signature != geist_main._local_agent_configuration_signature(
        _factory_config("artifact-b")
    )
    assert signature != geist_main._local_agent_configuration_signature(
        _factory_config("artifact-a", temperature=0.2)
    )
