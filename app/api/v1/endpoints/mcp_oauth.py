import secrets
from html import escape

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.api.v1.endpoints.mcp import _invalidate, _owned_server_or_404, _require_tools_operator
from app.models.mcp_oauth import (
    OAuthDisconnectResponse,
    OAuthStartRequest,
    OAuthStartResponse,
    OAuthStatus,
)
from app.security.operator import OperatorPrincipal
from app.services import mcp_oauth
from app.services.mcp_oauth_providers import CALLBACK_PATH, OAuthError, callback_uri
from app.services.mcp_oauth_store import CredentialStoreError


router = APIRouter()


@router.get("/servers/{server_id}/oauth", response_model=OAuthStatus)
def account_status(server_id: int, operator: OperatorPrincipal = Depends(_require_tools_operator)):
    server = _owned_server_or_404(server_id, operator.workspace_id)
    try:
        return mcp_oauth.status(server)
    except (OAuthError, CredentialStoreError) as error:
        raise HTTPException(409, str(error)) from error


@router.post("/servers/{server_id}/oauth/start", response_model=OAuthStartResponse)
def connect_account(
    server_id: int,
    body: OAuthStartRequest,
    request: Request,
    response: Response,
    operator: OperatorPrincipal = Depends(_require_tools_operator),
):
    server = _owned_server_or_404(server_id, operator.workspace_id)
    try:
        # The browser cookie must be set on the same origin used by the callback.
        public_origin = callback_uri().removesuffix(CALLBACK_PATH)
        origin = request.headers.get("origin")
        if origin and origin != public_origin:
            raise OAuthError("Open Geist at GEIST_PUBLIC_URL before connecting an account")
        browser_token = secrets.token_urlsafe(32)
        url = mcp_oauth.start(server, body.provider, browser_token)
        response.set_cookie(
            f"geist_mcp_oauth_{server_id}",
            browser_token,
            max_age=mcp_oauth.STATE_LIFETIME,
            httponly=True,
            secure=public_origin.startswith("https:"),
            samesite="lax",
            path=CALLBACK_PATH,
        )
        response.headers["Cache-Control"] = "no-store"
        _invalidate(server_id)
        return OAuthStartResponse(authorization_url=url)
    except (OAuthError, CredentialStoreError) as error:
        raise HTTPException(409, str(error)) from error


@router.get("/oauth/callback", include_in_schema=False)
async def authorization_callback(request: Request):
    params = request.query_params
    # Access logs must not retain authorization codes or state parameters.
    request.scope["query_string"] = b""
    state = params.get("state", "")
    server_part, separator, nonce = state.partition(".")
    server_id = (
        int(server_part)
        if server_part.isascii() and server_part.isdigit() and len(server_part) <= 10
        else 0
    )
    error_message = None
    try:
        if not server_id or not separator or len(nonce) != 43 or len(params.get("code", "")) > 8192:
            raise OAuthError("Invalid authorization callback; start the connection again")
        await run_in_threadpool(
            mcp_oauth.finish,
            server_id,
            state,
            request.cookies.get(f"geist_mcp_oauth_{server_id}", ""),
            params.get("code"),
            "error" in params,
        )
        _invalidate(server_id)
    except (OAuthError, CredentialStoreError) as error:
        error_message = str(error)
    response: Response
    if error_message:
        response = HTMLResponse(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Account connection</title><main><h1>Account not connected</h1>"
            f'<p>{escape(error_message)}</p><a href="/tools?tab=mcp">Return to Tools</a></main></html>',
            status_code=400,
        )
    else:
        response = RedirectResponse("/tools?mcp_oauth=connected", status_code=303)
    response.delete_cookie(f"geist_mcp_oauth_{server_id}", path=CALLBACK_PATH)
    response.headers.update(
        {
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        }
    )
    return response


@router.delete("/servers/{server_id}/oauth", response_model=OAuthDisconnectResponse)
def disconnect_account(
    server_id: int, operator: OperatorPrincipal = Depends(_require_tools_operator)
):
    server = _owned_server_or_404(server_id, operator.workspace_id)
    try:
        revoked = mcp_oauth.disconnect(server)
        _invalidate(server_id)
        return OAuthDisconnectResponse(revocation_complete=revoked)
    except (OAuthError, CredentialStoreError) as error:
        raise HTTPException(409, str(error)) from error
