from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api import utils as api_utils
from app.security import operator
from app.security.operator import (
    ALL_OPERATOR_CAPABILITIES,
    OPERATOR_AUTHENTICATION_SCHEME,
    OperatorCapability,
    OperatorPrincipal,
    get_operator_principal,
    require_operator_capability,
)


def request(
    *,
    client_host: str = "127.0.0.1",
    authorization: str | None = None,
    duplicate_authorization: bool = False,
) -> Request:
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("latin-1")))
        if duplicate_authorization:
            headers.append((b"authorization", authorization.encode("latin-1")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/api/v1/mcp/servers",
            "headers": headers,
            "client": (client_host, 43210),
            "server": ("127.0.0.1", 5001),
        }
    )


@pytest.fixture(autouse=True)
def default_workspace(monkeypatch):
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN_FILE", raising=False)
    monkeypatch.setattr(
        operator,
        "get_default_workspace",
        lambda: SimpleNamespace(workspace_id=41),
    )


@pytest.mark.parametrize("client_host", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_standalone_loopback_request_becomes_local_operator(client_host):
    principal = get_operator_principal(request(client_host=client_host))

    assert principal == OperatorPrincipal(
        subject="local-standalone",
        authentication_method="loopback",
        workspace_id=41,
        is_loopback=True,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )


def test_standalone_remote_request_is_rejected():
    with pytest.raises(HTTPException) as raised:
        get_operator_principal(request(client_host="192.0.2.10"))

    assert raised.value.status_code == 401
    assert raised.value.headers == {"WWW-Authenticate": OPERATOR_AUTHENTICATION_SCHEME}


def test_managed_runtime_requires_exact_wrapper_token(monkeypatch):
    token = "a" * 43
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", token)

    principal = get_operator_principal(
        request(
            client_host="192.0.2.10",
            authorization=f"{OPERATOR_AUTHENTICATION_SCHEME} {token}",
        )
    )

    assert principal.subject == "pitchblend-managed"
    assert principal.authentication_method == "wrapper-token"
    assert principal.is_local_operator is False
    assert principal.workspace_id == 41
    assert principal.is_loopback is False


@pytest.mark.parametrize(
    "authorization,duplicate",
    [
        (None, False),
        ("Bearer " + "a" * 43, False),
        ("GeistOperator wrong", False),
        ("GeistOperator " + "a" * 43, True),
    ],
)
def test_managed_runtime_rejects_missing_malformed_or_duplicate_tokens(
    monkeypatch, authorization, duplicate
):
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", "a" * 43)

    with pytest.raises(HTTPException) as raised:
        get_operator_principal(
            request(
                authorization=authorization,
                duplicate_authorization=duplicate,
            )
        )

    assert raised.value.status_code == 401


def test_invalid_configured_token_fails_closed(monkeypatch):
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", "too-short")

    with pytest.raises(HTTPException) as raised:
        get_operator_principal(request())

    assert raised.value.status_code == 503


def test_managed_runtime_accepts_token_from_private_file(monkeypatch, tmp_path):
    token = "f" * 43
    token_file = tmp_path / "operator-token"
    token_file.write_text(f"{token}\n", encoding="utf-8")
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN_FILE", str(token_file))

    principal = get_operator_principal(
        request(authorization=f"{OPERATOR_AUTHENTICATION_SCHEME} {token}")
    )

    assert principal.authentication_method == "local-token-file"
    assert principal.subject == "local-managed"
    assert principal.is_local_operator is True


def test_mismatched_environment_and_file_tokens_fail_closed(monkeypatch, tmp_path):
    token_file = tmp_path / "operator-token"
    token_file.write_text(f"{'f' * 43}\n", encoding="utf-8")
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", "e" * 43)
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN_FILE", str(token_file))

    with pytest.raises(HTTPException) as raised:
        get_operator_principal(request())

    assert raised.value.status_code == 503


def test_capability_dependency_rejects_underprivileged_principal():
    dependency = require_operator_capability(OperatorCapability.TOOLS_MANAGE)
    principal = OperatorPrincipal(
        subject="limited",
        authentication_method="test",
        workspace_id=41,
        is_loopback=True,
        capabilities=frozenset({OperatorCapability.WORKSPACE_READ}),
    )

    with pytest.raises(HTTPException) as raised:
        dependency(principal)

    assert raised.value.status_code == 403


def test_workspace_dependency_uses_principal_workspace(monkeypatch):
    workspace = SimpleNamespace(workspace_id=73)
    monkeypatch.setattr(
        api_utils,
        "get_default_workspace",
        lambda: workspace,
    )
    principal = OperatorPrincipal(
        subject="test",
        authentication_method="test",
        workspace_id=73,
        is_loopback=True,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )

    assert api_utils.get_current_workspace(principal) is workspace
