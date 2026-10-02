"""Explicit OAuth registrations bind credentials to reviewed MCP endpoints."""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


CALLBACK_PATH = "/api/v1/mcp/oauth/callback"
GMAIL_ENDPOINT = "https://gmailmcp.googleapis.com/mcp/v1"


class OAuthError(RuntimeError):
    pass


class OAuthProvider(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    label: str = Field(min_length=1, max_length=100)
    endpoint: str
    authorization_url: str
    token_url: str
    revocation_url: str | None = None
    scopes: list[str] = Field(min_length=1, max_length=30)
    client_id_env: str
    client_secret_env: str | None = None
    resource: str | None = None

    @field_validator("endpoint", "authorization_url", "token_url", "revocation_url", "resource")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.fragment
                or parsed.query
            ):
                raise ValueError(
                    "OAuth endpoints must be HTTPS URLs without credentials, query, or fragment"
                )
        return value

    @field_validator("client_id_env", "client_secret_env")
    @classmethod
    def validate_environment_name(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[A-Z_][A-Z0-9_]*", value) is None:
            raise ValueError("OAuth client environment variable name is invalid")
        return value

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, values: list[str]) -> list[str]:
        if any(
            not value or len(value) > 512 or any(c.isspace() for c in value) for value in values
        ):
            raise ValueError("OAuth scopes must be individual nonempty strings")
        return values

    def client_credentials(self) -> dict[str, str]:
        values = {"client_id": os.environ.get(self.client_id_env, "")}
        if not values["client_id"]:
            raise OAuthError(
                f"Configure {self.client_id_env} on the backend to connect {self.label}"
            )
        if self.client_secret_env:
            values["client_secret"] = os.environ.get(self.client_secret_env, "")
            if not values["client_secret"]:
                raise OAuthError(
                    f"Configure {self.client_secret_env} on the backend to connect {self.label}"
                )
        return values


def providers() -> list[OAuthProvider]:
    # The token URL is a public endpoint, not a password.
    google = OAuthProvider(  # nosec B106
        id="google",
        label="Google",
        endpoint=GMAIL_ENDPOINT,
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        revocation_url="https://oauth2.googleapis.com/revoke",
        scopes=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
        ],
        client_id_env="GEIST_GOOGLE_OAUTH_CLIENT_ID",
        client_secret_env="GEIST_GOOGLE_OAUTH_CLIENT_SECRET",
    )
    github = OAuthProvider(  # nosec B106
        id="github",
        label="GitHub",
        endpoint="https://api.githubcopilot.com/mcp/",
        authorization_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scopes=["repo"],
        client_id_env="GEIST_GITHUB_OAUTH_CLIENT_ID",
        client_secret_env="GEIST_GITHUB_OAUTH_CLIENT_SECRET",
    )
    configured = os.environ.get("GEIST_MCP_OAUTH_PROVIDERS", "[]")
    try:
        import json

        entries = json.loads(configured)
        if not isinstance(entries, list) or len(entries) > 30:
            raise ValueError()
        result = [google, github, *(OAuthProvider.model_validate(entry) for entry in entries)]
        if len({provider.id for provider in result}) != len(result):
            raise ValueError()
        return result
    except (ValueError, TypeError) as error:
        raise OAuthError(
            "GEIST_MCP_OAUTH_PROVIDERS contains invalid or duplicate provider registrations"
        ) from error


def get_provider(provider_id: str) -> OAuthProvider:
    for provider in providers():
        if provider.id == provider_id:
            return provider
    raise OAuthError("This OAuth provider is not configured on the backend")


def callback_uri() -> str:
    base = os.environ.get("GEIST_PUBLIC_URL", "http://localhost:3000").rstrip("/")
    parsed = urlsplit(base)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or (
            parsed.scheme != "https"
            and not (
                parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        )
    ):
        raise OAuthError("GEIST_PUBLIC_URL must be an HTTPS origin or a local HTTP origin")
    return base + CALLBACK_PATH
