import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from agents.agent_type import AgentType
from agents.model_load_status import ModelLoadStatusRegistry
from app import main as geist_main
from app.api.v1.endpoints import models as models_endpoint
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
    backend: str | None = None,
    device_ids: tuple[str, ...] = (),
) -> AgentFactoryConfig:
    device_config = {"artifact_id": artifact_id}
    if backend is not None:
        device_config["llama_backend"] = backend
        device_config["llama_gpu_device_ids"] = list(device_ids)
    return AgentFactoryConfig(
        agent_type="local",
        model=model,
        runner_type=runner_type,
        device_config=device_config,
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
    assert signature != geist_main._local_agent_configuration_signature(
        _factory_config("artifact-a", backend="cpu")
    )
    assert geist_main._local_agent_configuration_signature(
        _factory_config("artifact-a", backend="gpu", device_ids=("gpu-a",))
    ) != geist_main._local_agent_configuration_signature(
        _factory_config("artifact-a", backend="gpu", device_ids=("gpu-b",))
    )


def test_first_use_persists_effective_backend_and_caches_final_signature() -> None:
    saved_cache = {
        agent_type: geist_main.agent_cache[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    saved_signatures = {
        agent_type: geist_main._agent_cache_signatures[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    config = _factory_config("artifact-a", backend="auto")

    class DetectedAgent(RecordingLocalAgent):
        def runtime_selection(self):
            return "gpu", ("gpu-best",)

    try:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
        with (
            patch("app.main._get_local_agent_factory_config", return_value=config),
            patch(
                "app.main._create_local_agent",
                return_value=DetectedAgent("artifact-a", []),
            ),
            patch("app.main._llama_selection_managed_by_environment", return_value=False),
            patch(
                "app.main.get_default_workspace",
                return_value=type("User", (), {"workspace_id": 1})(),
            ),
            patch("app.main.UserSettingsService.persist_detected_llama_backend") as persist,
        ):
            persist.return_value = type(
                "Settings",
                (),
                {"llama_backend": "gpu", "llama_gpu_device_ids": ["gpu-best"]},
            )()
            geist_main.get_active_agent(AgentType.LLAMA)

        persist.assert_called_once_with(1, "gpu", ("gpu-best",))
        final_config = _factory_config("artifact-a", backend="gpu", device_ids=("gpu-best",))
        assert geist_main._agent_cache_signatures[AgentType.LLAMA] == (
            geist_main._local_agent_configuration_signature(final_config)
        )
    finally:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
            for agent_type in geist_main._LOCAL_AGENT_TYPES:
                geist_main.agent_cache[agent_type] = saved_cache[agent_type]
                geist_main._agent_cache_signatures[agent_type] = saved_signatures[agent_type]


def test_first_use_persists_clean_cpu_detection() -> None:
    saved_cache = {
        agent_type: geist_main.agent_cache[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    saved_signatures = {
        agent_type: geist_main._agent_cache_signatures[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    config = _factory_config("artifact-a", backend="auto")

    class DetectedCpuAgent(RecordingLocalAgent):
        def runtime_selection(self):
            return "cpu", ()

        def runtime_selection_detection_error(self):
            return None

    try:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
        with (
            patch("app.main._get_local_agent_factory_config", return_value=config),
            patch(
                "app.main._create_local_agent",
                return_value=DetectedCpuAgent("artifact-a", []),
            ),
            patch("app.main._llama_selection_managed_by_environment", return_value=False),
            patch(
                "app.main.get_default_workspace",
                return_value=type("User", (), {"workspace_id": 1})(),
            ),
            patch("app.main.UserSettingsService.persist_detected_llama_backend") as persist,
        ):
            persist.return_value = type(
                "Settings",
                (),
                {"llama_backend": "cpu", "llama_gpu_device_ids": []},
            )()
            geist_main.get_active_agent(AgentType.LLAMA)

        persist.assert_called_once_with(1, "cpu", ())
        final_config = _factory_config("artifact-a", backend="cpu")
        assert geist_main._agent_cache_signatures[AgentType.LLAMA] == (
            geist_main._local_agent_configuration_signature(final_config)
        )
    finally:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
            for agent_type in geist_main._LOCAL_AGENT_TYPES:
                geist_main.agent_cache[agent_type] = saved_cache[agent_type]
                geist_main._agent_cache_signatures[agent_type] = saved_signatures[agent_type]


def test_failed_auto_cpu_discovery_remains_pending_without_persistence(monkeypatch, caplog) -> None:
    monkeypatch.setattr(geist_main, "_pending_detection_warning", None)
    saved_cache = {
        agent_type: geist_main.agent_cache[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    saved_signatures = {
        agent_type: geist_main._agent_cache_signatures[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }

    class FailedDetectionCpuAgent(RecordingLocalAgent):
        def runtime_selection(self):
            return "cpu", ()

        def runtime_selection_detection_error(self):
            return "llama-server --list-devices timed out"

    try:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
        with (
            patch(
                "app.main._get_local_agent_factory_config",
                side_effect=lambda: _factory_config("artifact-a", backend="auto"),
            ),
            patch(
                "app.main._create_local_agent",
                return_value=FailedDetectionCpuAgent("artifact-a", []),
            ) as create,
            patch("app.main._llama_selection_managed_by_environment", return_value=False),
            patch("app.main.get_default_workspace") as get_user,
            patch("app.main.UserSettingsService.persist_detected_llama_backend") as persist,
        ):
            first = geist_main.get_active_agent(AgentType.LLAMA)
            second = geist_main.get_active_agent(AgentType.LLAMA)

        assert second is first
        assert create.call_count == 1
        get_user.assert_not_called()
        persist.assert_not_called()
        assert caplog.text.count("First-use compute detection remains pending") == 1
        automatic_config = _factory_config("artifact-a", backend="auto")
        assert geist_main._agent_cache_signatures[AgentType.LLAMA] == (
            geist_main._local_agent_configuration_signature(automatic_config)
        )
    finally:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
            for agent_type in geist_main._LOCAL_AGENT_TYPES:
                geist_main.agent_cache[agent_type] = saved_cache[agent_type]
                geist_main._agent_cache_signatures[agent_type] = saved_signatures[agent_type]


def test_cached_auto_agent_retries_backend_persistence_after_transient_failure() -> None:
    saved_cache = {
        agent_type: geist_main.agent_cache[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    saved_signatures = {
        agent_type: geist_main._agent_cache_signatures[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }

    class DetectedAgent(RecordingLocalAgent):
        def runtime_selection(self):
            return "gpu", ("gpu-best",)

    persisted = type(
        "Settings",
        (),
        {"llama_backend": "gpu", "llama_gpu_device_ids": ["gpu-best"]},
    )()
    try:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
        with (
            patch(
                "app.main._get_local_agent_factory_config",
                side_effect=lambda: _factory_config("artifact-a", backend="auto"),
            ),
            patch(
                "app.main._create_local_agent",
                return_value=DetectedAgent("artifact-a", []),
            ) as create,
            patch("app.main._llama_selection_managed_by_environment", return_value=False),
            patch(
                "app.main.get_default_workspace",
                return_value=type("User", (), {"workspace_id": 1})(),
            ),
            patch(
                "app.main.UserSettingsService.persist_detected_llama_backend",
                side_effect=[RuntimeError("database unavailable"), persisted],
            ) as persist,
        ):
            first = geist_main.get_active_agent(AgentType.LLAMA)
            second = geist_main.get_active_agent(AgentType.LLAMA)

        assert second is first
        assert create.call_count == 1
        assert persist.call_count == 2
        final_config = _factory_config("artifact-a", backend="gpu", device_ids=("gpu-best",))
        assert geist_main._agent_cache_signatures[AgentType.LLAMA] == (
            geist_main._local_agent_configuration_signature(final_config)
        )
    finally:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
            for agent_type in geist_main._LOCAL_AGENT_TYPES:
                geist_main.agent_cache[agent_type] = saved_cache[agent_type]
                geist_main._agent_cache_signatures[agent_type] = saved_signatures[agent_type]


@pytest.mark.parametrize("replace_during_persistence", [False, True])
def test_cached_persistence_releases_lock_and_cannot_resurrect_agent(
    monkeypatch, replace_during_persistence
) -> None:
    config = _factory_config("cached", backend="auto")
    agent = RecordingLocalAgent("cached", [])
    replacement = RecordingLocalAgent("replacement", [])
    cache = dict(geist_main.agent_cache)
    signatures = dict(geist_main._agent_cache_signatures)
    for local_type in geist_main._LOCAL_AGENT_TYPES:
        cache[local_type] = agent
        signatures[local_type] = geist_main._local_agent_configuration_signature(config)
    monkeypatch.setattr(geist_main, "agent_cache", cache)
    monkeypatch.setattr(geist_main, "_agent_cache_signatures", signatures)
    monkeypatch.setattr(geist_main, "_get_local_agent_factory_config", lambda: config)
    acquired = []

    def persist(*_args):
        def inspect_lock():
            held = geist_main._agent_cache_lock.acquire(timeout=0.5)
            acquired.append(held)
            if held:
                try:
                    if replace_during_persistence:
                        geist_main._set_local_agent_cache(replacement, "replacement")
                finally:
                    geist_main._agent_cache_lock.release()

        thread = threading.Thread(target=inspect_lock)
        thread.start()
        thread.join(timeout=1)
        return "persisted"

    monkeypatch.setattr(geist_main, "_persist_first_use_llama_backend", persist)
    assert geist_main.get_active_agent(AgentType.LLAMA) is agent
    assert acquired == [True]
    expected = replacement if replace_during_persistence else agent
    assert all(cache[local_type] is expected for local_type in geist_main._LOCAL_AGENT_TYPES)


def test_explicit_binary_does_not_claim_persisted_gpu_acceleration(monkeypatch):
    monkeypatch.setenv("GEIST_LLAMA_SERVER_PATH", "/operator/server")
    monkeypatch.delenv("GEIST_LLAMA_ACCELERATION", raising=False)
    assert geist_main._llama_acceleration("llama_server", "gpu") is None


def test_acceleration_prefers_matching_live_runtime_and_ignores_invalid_override(monkeypatch):
    from types import SimpleNamespace

    from agents.architectures import llama_server_process

    monkeypatch.delenv("GEIST_LLAMA_SERVER_PATH", raising=False)
    monkeypatch.setenv("GEIST_LLAMA_ACCELERATION", "cuda")
    monkeypatch.setattr(
        llama_server_process,
        "get_llama_server_manager",
        lambda: SimpleNamespace(
            public_status=lambda: {"status": "ready", "model_id": "loaded", "backend": "vulkan"},
        ),
    )
    assert geist_main._llama_acceleration("llama_server", None, "loaded") == "vulkan"
    assert geist_main._llama_acceleration("llama_server", "cpu", "other") == "cpu"


def test_concurrent_manual_choice_is_not_cached_as_the_auto_runtime() -> None:
    saved_cache = {
        agent_type: geist_main.agent_cache[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    saved_signatures = {
        agent_type: geist_main._agent_cache_signatures[agent_type]
        for agent_type in geist_main._LOCAL_AGENT_TYPES
    }
    config = _factory_config("artifact-a", backend="auto")

    class DetectedGpuAgent(RecordingLocalAgent):
        def runtime_selection(self):
            return "gpu", ("gpu-best",)

    try:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
        with (
            patch("app.main._get_local_agent_factory_config", return_value=config),
            patch(
                "app.main._create_local_agent",
                return_value=DetectedGpuAgent("artifact-a", []),
            ),
            patch("app.main._llama_selection_managed_by_environment", return_value=False),
            patch(
                "app.main.get_default_workspace",
                return_value=type("User", (), {"workspace_id": 1})(),
            ),
            patch(
                "app.main.UserSettingsService.persist_detected_llama_backend",
                return_value=type(
                    "Settings",
                    (),
                    {"llama_backend": "cpu", "llama_gpu_device_ids": []},
                )(),
            ),
        ):
            geist_main.get_active_agent(AgentType.LLAMA)

        actual_gpu_signature = geist_main._local_agent_configuration_signature(
            _factory_config("artifact-a", backend="gpu", device_ids=("gpu-best",))
        )
        persisted_cpu_signature = geist_main._local_agent_configuration_signature(
            _factory_config("artifact-a", backend="cpu")
        )
        assert geist_main._agent_cache_signatures[AgentType.LLAMA] == actual_gpu_signature
        assert actual_gpu_signature != persisted_cpu_signature
    finally:
        with geist_main._agent_cache_lock:
            geist_main._clear_local_agent_cache()
            for agent_type in geist_main._LOCAL_AGENT_TYPES:
                geist_main.agent_cache[agent_type] = saved_cache[agent_type]
                geist_main._agent_cache_signatures[agent_type] = saved_signatures[agent_type]


def test_stalled_local_cleanup_does_not_hold_shared_cache_lock(monkeypatch):
    saved_cache = dict(geist_main.agent_cache)
    saved_signatures = dict(geist_main._agent_cache_signatures)
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()

    class StalledAgent:
        def phase_out(self):
            cleanup_started.set()
            assert release_cleanup.wait(5)

    old = StalledAgent()
    new = object()
    online = object()
    try:
        with geist_main._agent_cache_lock:
            geist_main._set_local_agent_cache(old, "old-signature")
            geist_main.agent_cache[AgentType.GPT4AGENT] = online
        monkeypatch.setattr(
            geist_main, "_get_local_agent_factory_config", lambda: _factory_config("new")
        )
        monkeypatch.setattr(geist_main, "_create_local_agent", lambda _: new)
        with ThreadPoolExecutor(max_workers=2) as callers:
            switching = callers.submit(geist_main.get_or_create_agent, AgentType.LOCALAGENT)
            try:
                assert cleanup_started.wait(2)
                assert (
                    callers.submit(geist_main.get_or_create_agent, AgentType.GPT4AGENT).result(
                        timeout=2
                    )
                    is online
                )
                with pytest.raises(RuntimeError, match="loading or switching"):
                    geist_main.get_or_create_agent(AgentType.LOCALAGENT)
            finally:
                release_cleanup.set()
            assert switching.result(timeout=2) is new
        assert not geist_main._local_agent_creation_lock.locked()
    finally:
        release_cleanup.set()
        with geist_main._agent_cache_lock:
            geist_main.agent_cache.update(saved_cache)
            geist_main._agent_cache_signatures.update(saved_signatures)


@pytest.mark.parametrize("ready_before_publishing", [False, True])
@pytest.mark.parametrize("owner_is_readiness", [False, True])
@pytest.mark.parametrize("load_fails", [False, True])
@pytest.mark.parametrize("configured_model", ["target", ""])
def test_duplicate_readiness_start_follows_inflight_load_to_terminal_state(
    monkeypatch, ready_before_publishing, owner_is_readiness, load_fails, configured_model
):
    model_id = configured_model or geist_main.DEFAULT_LOCAL_MODEL
    statuses = ModelLoadStatusRegistry()
    monkeypatch.setattr(models_endpoint, "model_load_status_registry", statuses)
    monkeypatch.setattr(geist_main, "model_load_status_registry", statuses)
    monkeypatch.setattr(geist_main, "agent_cache", {key: None for key in geist_main.agent_cache})
    monkeypatch.setattr(
        geist_main, "_agent_cache_signatures", {key: None for key in geist_main.agent_cache}
    )
    monkeypatch.setattr(
        geist_main,
        "_get_local_agent_factory_config",
        lambda: _factory_config("a", model=configured_model),
    )
    entered, release = threading.Event(), threading.Event()
    created = []

    def create(_):
        created.append(True)
        if ready_before_publishing:
            statuses.mark_ready(model_id)
        entered.set()
        assert release.wait(5)
        if load_fails:
            raise ValueError("Initialization failed before runner.load")
        return object()

    monkeypatch.setattr(geist_main, "_create_local_agent", create)
    statuses.mark_loading(model_id, "Loading")
    with ThreadPoolExecutor(max_workers=2) as requests:
        if owner_is_readiness:
            first = requests.submit(models_endpoint._initialize_configured_local_runtime, model_id)
        else:
            first = requests.submit(geist_main.get_or_create_agent, AgentType.LOCALAGENT)
        try:
            assert entered.wait(2)
            statuses.mark_loading(model_id, "Repeated readiness request")
            second = requests.submit(models_endpoint._initialize_configured_local_runtime, model_id)
            second.result(timeout=2)
            assert statuses.get(model_id).state == "loading"
        finally:
            release.set()
        if load_fails and not owner_is_readiness:
            with pytest.raises(ValueError, match="Initialization failed before runner.load"):
                first.result(timeout=2)
        else:
            first.result(timeout=2)
    assert statuses.get(model_id).state == ("failed" if load_fails else "ready")
    if load_fails:
        assert statuses.get(model_id).detail == "Initialization failed before runner.load"
    assert len(created) == 1
    assert geist_main._local_agent_loading_model_id is None
    assert not geist_main._local_agent_creation_lock.locked()


@pytest.mark.parametrize("state", ["ready", "failed"])
def test_busy_readiness_preserves_load_owner_terminal_result(monkeypatch, state):
    statuses = ModelLoadStatusRegistry()
    monkeypatch.setattr(models_endpoint, "model_load_status_registry", statuses)
    if state == "ready":
        statuses.mark_ready("target")
    else:
        statuses.mark_failed("target", "Real load failure")
    before = statuses.get("target")

    def busy(_):
        raise geist_main.LocalModelBusyError("target")

    monkeypatch.setattr(geist_main, "get_active_agent", busy)
    models_endpoint._initialize_configured_local_runtime("target")
    assert statuses.get("target") == before


def test_other_model_busy_does_not_leave_unowned_loading_status(monkeypatch):
    statuses = ModelLoadStatusRegistry()
    statuses.mark_loading("requested", "Loading")
    statuses.mark_loading("other", "Loading")
    monkeypatch.setattr(models_endpoint, "model_load_status_registry", statuses)

    def busy(_):
        raise geist_main.LocalModelBusyError("other")

    monkeypatch.setattr(geist_main, "get_active_agent", busy)
    models_endpoint._initialize_configured_local_runtime("requested")
    assert statuses.get("requested").state == "failed"
    assert statuses.get("other").state == "loading"
