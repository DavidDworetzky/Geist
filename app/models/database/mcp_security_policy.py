"""Per-user policy for inspecting untrusted MCP boundaries."""

import datetime
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String

from app.models.database.database import Base, SessionLocal


class McpSecurityPolicy(Base):
    __tablename__ = "mcp_security_policy"

    mcp_security_policy_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False, unique=True)
    enabled = Column(Boolean, nullable=False, default=True)
    inspect_tool_metadata = Column(Boolean, nullable=False, default=True)
    inspect_outbound_arguments = Column(Boolean, nullable=False, default=True)
    inspect_inbound_results = Column(Boolean, nullable=False, default=True)
    deterministic_scanner = Column(Boolean, nullable=False, default=True)
    model_mode = Column(String, nullable=False, default="mirror")
    create_date = Column(DateTime, default=datetime.datetime.utcnow)
    update_date = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )


@dataclass
class McpSecurityPolicyModel:
    mcp_security_policy_id: int
    user_id: int
    enabled: bool
    inspect_tool_metadata: bool
    inspect_outbound_arguments: bool
    inspect_inbound_results: bool
    deterministic_scanner: bool
    model_mode: str
    create_date: datetime.datetime
    update_date: datetime.datetime


_MUTABLE_FIELDS = (
    "enabled",
    "inspect_tool_metadata",
    "inspect_outbound_arguments",
    "inspect_inbound_results",
    "deterministic_scanner",
)


def _to_model(policy: McpSecurityPolicy) -> McpSecurityPolicyModel:
    return McpSecurityPolicyModel(
        mcp_security_policy_id=policy.mcp_security_policy_id,
        user_id=policy.user_id,
        enabled=bool(policy.enabled),
        inspect_tool_metadata=bool(policy.inspect_tool_metadata),
        inspect_outbound_arguments=bool(policy.inspect_outbound_arguments),
        inspect_inbound_results=bool(policy.inspect_inbound_results),
        deterministic_scanner=bool(policy.deterministic_scanner),
        model_mode=policy.model_mode or "mirror",
        create_date=policy.create_date,
        update_date=policy.update_date,
    )


def get_or_create_mcp_security_policy(user_id: int) -> McpSecurityPolicyModel:
    with SessionLocal() as session:
        policy = session.query(McpSecurityPolicy).filter_by(user_id=user_id).first()
        if policy is None:
            policy = McpSecurityPolicy(user_id=user_id)
            session.add(policy)
            session.commit()
            session.refresh(policy)
        return _to_model(policy)


def update_mcp_security_policy(
    user_id: int,
    updates: dict[str, Any],
) -> McpSecurityPolicyModel:
    with SessionLocal() as session:
        policy = session.query(McpSecurityPolicy).filter_by(user_id=user_id).first()
        if policy is None:
            policy = McpSecurityPolicy(user_id=user_id)
            session.add(policy)
        for field in _MUTABLE_FIELDS:
            if field in updates:
                setattr(policy, field, updates[field])
        policy.update_date = datetime.datetime.utcnow()
        session.commit()
        session.refresh(policy)
        return _to_model(policy)
