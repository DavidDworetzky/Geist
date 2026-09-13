import datetime
import threading
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from app.api.v1.endpoints.user_settings import get_current_workspace, router
from app.models.user_settings import UserSettingsResponse
from app.services.user_settings_service import UserSettingsService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({}, 200),
        ({"llama_allow_system_ram": False}, 200),
        ({"llama_allow_system_ram": True}, 200),
        ({"llama_allow_system_ram": None}, 422),
    ],
)
async def test_ram_preference_accepts_omission_and_booleans_but_rejects_null(
    payload, expected_status
):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/user-settings")
    app.dependency_overrides[get_current_workspace] = lambda: SimpleNamespace(workspace_id=1)
    response = UserSettingsResponse(
        user_settings_id=1,
        user_id=1,
        create_date=datetime.datetime.now(datetime.UTC),
        update_date=datetime.datetime.now(datetime.UTC),
    )
    with patch.object(
        UserSettingsService, "update_workspace_settings_by_id", return_value=response
    ) as write:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.put("/api/v1/user-settings/", json=payload)
    assert result.status_code == expected_status
    if expected_status == 422:
        write.assert_not_called()
        assert result.json()["detail"][0]["loc"] == ["body", "llama_allow_system_ram"]
    else:
        write.assert_called_once()
        assert write.call_args.args[1].model_dump(exclude_unset=True) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "timeout"])
async def test_compute_probe_failure_is_actionable_validation_error(failure) -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/user-settings")
    app.dependency_overrides[get_current_workspace] = lambda: SimpleNamespace(workspace_id=1)
    current = SimpleNamespace(
        default_local_model="old/model", llama_backend="cpu", llama_gpu_device_ids=[]
    )
    with (
        patch("app.services.user_settings_service.get_user_settings", return_value=current),
        patch("app.services.user_settings_service.update_user_settings") as persist,
        patch("agents.architectures.llama_devices.get_llama_device_service") as discovery,
    ):
        if failure == "exception":
            discovery.return_value.inventory.side_effect = RuntimeError("private failure detail")
        else:
            discovery.return_value.inventory.return_value = SimpleNamespace(
                managed_by_environment=False,
                available=True,
                devices=(),
                selection_detection_error="timed out",
            )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.put(
                "/api/v1/user-settings/",
                json={"llama_backend": "gpu", "llama_gpu_device_ids": ["gpu-1"]},
            )
    assert result.status_code == 422
    assert "retry saving compute settings" in result.text
    assert "private failure detail" not in result.text
    persist.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/v1/user-settings/"])
async def test_compute_update_routes_run_service_off_event_loop(path: str) -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/user-settings")
    app.dependency_overrides[get_current_workspace] = lambda: SimpleNamespace(workspace_id=1)
    response = UserSettingsResponse(
        user_settings_id=1,
        user_id=1,
        create_date=datetime.datetime.now(datetime.UTC),
        update_date=datetime.datetime.now(datetime.UTC),
    )
    event_loop_thread = threading.get_ident()
    service_threads: list[int] = []

    def update_settings(*args, **kwargs):
        service_threads.append(threading.get_ident())
        return response

    transport = httpx.ASGITransport(app=app)
    with patch.object(
        UserSettingsService,
        "update_workspace_settings_by_id",
        side_effect=update_settings,
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            result = await client.put(path, json={"llama_backend": "cpu"})

    assert result.status_code == 200
    assert service_threads and service_threads[0] != event_loop_thread


def test_settings_routes_cannot_select_a_numeric_owner():
    paths = {route.path for route in router.routes}
    assert paths == {"/", "/reset", "/agent-config/preview"}


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [[" "], ["x" * 257], ["web.search"] * 257])
async def test_permission_allowlist_dto_limits_reject_before_persistence(names):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/user-settings")
    app.dependency_overrides[get_current_workspace] = lambda: SimpleNamespace(workspace_id=1)
    with patch.object(UserSettingsService, "update_workspace_settings_by_id") as write:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.put(
                "/api/v1/user-settings/", json={"agent_permissions": {"always_allow": names}}
            )
    assert result.status_code == 422
    write.assert_not_called()
