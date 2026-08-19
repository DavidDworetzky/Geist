"""Security policy and blocked-content reveal endpoints."""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from app.models.database.geist_user import get_default_user
from app.models.database.mcp_security_policy import (
    get_or_create_mcp_security_policy,
    update_mcp_security_policy,
)
from app.services.mcp_security import blocked_originals
from app.services.mcp_tool_source import get_mcp_tool_source


router = APIRouter()


class SecurityPolicyResponse(BaseModel):
    mcp_security_policy_id: int
    user_id: int
    enabled: bool
    inspect_tool_metadata: bool
    inspect_outbound_arguments: bool
    inspect_inbound_results: bool
    deterministic_scanner: bool
    model_mode: str
    create_date: datetime
    update_date: datetime


class SecurityPolicyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    inspect_tool_metadata: bool | None = None
    inspect_outbound_arguments: bool | None = None
    inspect_inbound_results: bool | None = None
    deterministic_scanner: bool | None = None


class BlockedOriginalResponse(BaseModel):
    blocked_id: str
    content: str


@router.get("/policy", response_model=SecurityPolicyResponse)
async def get_policy():
    user = get_default_user()
    return SecurityPolicyResponse(**get_or_create_mcp_security_policy(user.user_id).__dict__)


@router.put("/policy", response_model=SecurityPolicyResponse)
async def update_policy(request: SecurityPolicyUpdate):
    user = get_default_user()
    policy = update_mcp_security_policy(
        user.user_id,
        request.model_dump(exclude_unset=True),
    )
    get_mcp_tool_source().invalidate()
    return SecurityPolicyResponse(**policy.__dict__)


@router.get("/blocked/{blocked_id}", response_model=BlockedOriginalResponse)
async def reveal_blocked_original(blocked_id: str):
    user = get_default_user()
    content = blocked_originals.get(user.user_id, blocked_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Blocked content not found")
    return BlockedOriginalResponse(blocked_id=blocked_id, content=content)
