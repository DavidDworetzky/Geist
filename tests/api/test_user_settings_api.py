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
