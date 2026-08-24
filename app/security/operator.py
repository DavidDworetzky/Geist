from __future__ import annotations

import ipaddress
import os
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from fastapi import Depends, HTTPException, Request, status

from app.models.database.geist_user import get_default_workspace


OPERATOR_TOKEN_ENVIRONMENT_VARIABLE = "GEIST_OPERATOR_TOKEN"
OPERATOR_TOKEN_FILE_ENVIRONMENT_VARIABLE = "GEIST_OPERATOR_TOKEN_FILE"
OPERATOR_AUTHENTICATION_SCHEME = "GeistOperator"
MINIMUM_OPERATOR_TOKEN_LENGTH = 32


class OperatorCapability(str, Enum):
    WORKSPACE_READ = "workspace.read"
    WORKSPACE_WRITE = "workspace.write"
    TOOLS_MANAGE = "tools.manage"
    TOOLS_EXECUTE = "tools.execute"


ALL_OPERATOR_CAPABILITIES = frozenset(OperatorCapability)


@dataclass(frozen=True)
class OperatorPrincipal:
    subject: str
    authentication_method: str
    workspace_id: int
    is_loopback: bool
    capabilities: frozenset[OperatorCapability]
    controller_node_id: str | None = None
    target_node_id: str | None = None
    audience: str | None = None
    expires_at: datetime | None = None
    credential_id: str | None = None

    def has_capability(self, capability: OperatorCapability) -> bool:
        return capability in self.capabilities

    @property
    def is_local_operator(self) -> bool:
        return self.is_loopback or self.authentication_method == "local-token-file"


def get_operator_principal(request: Request) -> OperatorPrincipal:
    """Authenticate the process operating this local Geist workspace."""
    client_host = request.client.host if request.client else None
    is_loopback = bool(client_host and _is_loopback_address(client_host))
    configured_token, token_source = _configured_operator_token()
    if configured_token:
        if (
            len(configured_token) < MINIMUM_OPERATOR_TOKEN_LENGTH
            or configured_token.strip() != configured_token
            or any(character.isspace() for character in configured_token)
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Geist operator authentication is misconfigured",
            )
        supplied_token = _operator_token_from_request(request)
        if supplied_token is None or not secrets.compare_digest(
            supplied_token, configured_token
        ):
            raise _not_authenticated()
        authentication_method = token_source
        subject = (
            "pitchblend-managed" if token_source == "wrapper-token" else "local-managed"
        )
    else:
        if not is_loopback:
            raise _not_authenticated()
        authentication_method = "loopback"
        subject = "local-standalone"

    workspace = get_default_workspace()
    return OperatorPrincipal(
        subject=subject,
        authentication_method=authentication_method,
        workspace_id=workspace.workspace_id,
        is_loopback=is_loopback,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )


def require_operator_capability(
    capability: OperatorCapability,
) -> Callable[[OperatorPrincipal], OperatorPrincipal]:
    def dependency(
        principal: OperatorPrincipal = Depends(get_operator_principal),
    ) -> OperatorPrincipal:
        if not principal.has_capability(capability):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator capability is required",
            )
        return principal

    return dependency


def _operator_token_from_request(request: Request) -> str | None:
    authorization_headers: list[str] = [
        value.decode("latin-1").strip()
        for name, value in request.scope.get("headers", [])
        if name.lower() == b"authorization"
    ]
    if len(authorization_headers) != 1:
        return None
    scheme, separator, token = authorization_headers[0].partition(" ")
    if (
        scheme != OPERATOR_AUTHENTICATION_SCHEME
        or separator != " "
        or not token
        or token.strip() != token
        or any(character.isspace() for character in token)
    ):
        return None
    return token


def _configured_operator_token() -> tuple[str | None, str]:
    environment_token = os.getenv(OPERATOR_TOKEN_ENVIRONMENT_VARIABLE)
    token_file = os.getenv(OPERATOR_TOKEN_FILE_ENVIRONMENT_VARIABLE)
    file_token: str | None = None
    if token_file:
        try:
            with Path(token_file).open("rb") as handle:
                payload = handle.read(4097)
        except OSError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Geist operator authentication is misconfigured",
            ) from error
        if len(payload) > 4096:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Geist operator authentication is misconfigured",
            )
        try:
            file_token = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Geist operator authentication is misconfigured",
            ) from error
        if file_token.endswith("\n"):
            file_token = file_token[:-1]
        if not file_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Geist operator authentication is misconfigured",
            )
    if environment_token and file_token and not secrets.compare_digest(
        environment_token,
        file_token,
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Geist operator authentication is misconfigured",
        )
    if environment_token:
        return environment_token, "wrapper-token"
    if file_token:
        return file_token, "local-token-file"
    return None, "loopback"


def _is_loopback_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return value.rstrip(".").lower() == "localhost"
    if address.is_loopback:
        return True
    return bool(
        isinstance(address, ipaddress.IPv6Address)
        and address.ipv4_mapped is not None
        and address.ipv4_mapped.is_loopback
    )


def _not_authenticated() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Operator authentication is required",
        headers={"WWW-Authenticate": OPERATOR_AUTHENTICATION_SCHEME},
    )
