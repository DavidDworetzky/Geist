import pytest
from fastapi import HTTPException

import app.api.utils as api_utils
from app.models.database.geist_user import WorkspaceModel
from app.security.operator import ALL_OPERATOR_CAPABILITIES, OperatorPrincipal


def principal(workspace_id: int) -> OperatorPrincipal:
    return OperatorPrincipal(
        subject="test",
        authentication_method="test",
        workspace_id=workspace_id,
        is_loopback=True,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )


def test_current_workspace_returns_matching_workspace(monkeypatch):
    workspace = WorkspaceModel(
        workspace_id=7,
        workspace_key="default",
        display_name="Local Workspace",
    )
    monkeypatch.setattr(api_utils, "get_default_workspace", lambda: workspace)

    result = api_utils.get_current_workspace(principal(7))

    assert result == workspace


def test_current_workspace_rejects_mismatched_principal(monkeypatch):
    workspace = WorkspaceModel(
        workspace_id=7,
        workspace_key="default",
        display_name="Local Workspace",
    )
    monkeypatch.setattr(api_utils, "get_default_workspace", lambda: workspace)

    with pytest.raises(HTTPException) as raised:
        api_utils.get_current_workspace(principal(8))

    assert raised.value.status_code == 403


def test_current_workspace_does_not_mask_database_failures(monkeypatch):
    def fail_lookup():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(api_utils, "get_default_workspace", fail_lookup)

    with pytest.raises(RuntimeError, match="database unavailable"):
        api_utils.get_current_workspace(principal(7))
