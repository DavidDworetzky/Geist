import pytest
from fastapi import HTTPException

import app.api.utils as api_utils
from app.models.database.geist_user import WorkspaceModel


@pytest.mark.asyncio
async def test_authenticated_workspace_returns_matching_workspace(monkeypatch):
    workspace = WorkspaceModel(
        workspace_id=7,
        workspace_key="default",
        display_name="Local Workspace",
    )
    monkeypatch.setattr(api_utils, "get_default_workspace", lambda: workspace)

    result = await api_utils.get_authenticated_workspace("test_token_7")

    assert result == workspace


@pytest.mark.asyncio
async def test_authenticated_workspace_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(
        api_utils,
        "get_default_workspace",
        lambda: pytest.fail("invalid credentials must not query the database"),
    )

    with pytest.raises(HTTPException) as raised:
        await api_utils.get_authenticated_workspace("not-a-test-token")

    assert raised.value.status_code == 401


@pytest.mark.asyncio
async def test_authenticated_workspace_rejects_mismatched_workspace(monkeypatch):
    workspace = WorkspaceModel(
        workspace_id=7,
        workspace_key="default",
        display_name="Local Workspace",
    )
    monkeypatch.setattr(api_utils, "get_default_workspace", lambda: workspace)

    with pytest.raises(HTTPException) as raised:
        await api_utils.get_authenticated_workspace("test_token_8")

    assert raised.value.status_code == 401


@pytest.mark.asyncio
async def test_authenticated_workspace_does_not_mask_database_failures(monkeypatch):
    def fail_lookup():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(api_utils, "get_default_workspace", fail_lookup)

    with pytest.raises(RuntimeError, match="database unavailable"):
        await api_utils.get_authenticated_workspace("test_token_7")
