import datetime
import threading
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from app.api.v1.endpoints.user_settings import get_current_user, router
from app.models.user_settings import UserSettingsResponse
from app.services.user_settings_service import UserSettingsService


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get", "put"])
async def test_by_id_settings_reject_other_owners_before_service_access(method):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/user-settings")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=1)
    with (
        patch.object(UserSettingsService, "get_user_settings_by_id") as read,
        patch.object(UserSettingsService, "update_user_settings_by_id") as write,
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            options = (
                {"json": {"agent_permissions": {"mode": "auto_approve"}}} if method == "put" else {}
            )
            result = await client.request(method, "/api/v1/user-settings/2", **options)
    assert result.status_code == 403
    read.assert_not_called()
    write.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [[" "], ["x" * 257], ["web.search"] * 257])
async def test_permission_allowlist_limits_reject_before_persistence(names):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/user-settings")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=1)
    with patch.object(UserSettingsService, "update_user_settings_by_id") as write:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.put(
                "/api/v1/user-settings/", json={"agent_permissions": {"always_allow": names}}
            )
    assert result.status_code == 422
    write.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/v1/user-settings/", "/api/v1/user-settings/1"])
async def test_compute_update_routes_run_service_off_event_loop(path: str) -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/user-settings")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=1)
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
        "update_user_settings_by_id",
        side_effect=update_settings,
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            result = await client.put(path, json={"llama_backend": "cpu"})

    assert result.status_code == 200
    assert service_threads and service_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_reset_restores_balanced_permissions_and_removes_grants():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/user-settings")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=1)
    response = UserSettingsResponse(
        user_settings_id=1,
        user_id=1,
        create_date=datetime.datetime.now(datetime.UTC),
        update_date=datetime.datetime.now(datetime.UTC),
    )
    with patch.object(
        UserSettingsService, "update_user_settings_by_id", return_value=response
    ) as write:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.post("/api/v1/user-settings/reset")
    assert result.status_code == 200
    updates = write.call_args.args[1]
    assert updates.agent_permissions.mode == "default"
    assert updates.agent_permissions.always_allow == []
