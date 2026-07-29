"""Load and normalize per-user agent permission settings.

Persisted on ``user_settings.agent_permissions`` as
``{"mode": <permission mode>, "always_allow": [tool names]}``. The mode is the
Hermes-style approval posture (auto-approve everything, require approval for
everything, or defer to per-tool flags) and ``always_allow`` is the standing
per-tool allowlist that skips approval even under the strict mode.
"""

import logging
from dataclasses import dataclass, field

from agents.models.tool_calling import (
    PERMISSION_MODE_DEFAULT,
    VALID_PERMISSION_MODES,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentPermissions:
    mode: str = PERMISSION_MODE_DEFAULT
    always_allow: frozenset[str] = field(default_factory=frozenset)


def normalize_agent_permissions(raw: object) -> dict:
    """Coerce a stored/submitted agent_permissions value into canonical shape.

    Raises ValueError on an unknown mode or a non-list allowlist so API
    updates fail with 422 instead of persisting settings the gate would then
    have to guess about. Unknown extra keys are dropped.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("agent_permissions must be an object")

    mode = raw.get("mode", PERMISSION_MODE_DEFAULT)
    if not isinstance(mode, str) or mode not in VALID_PERMISSION_MODES:
        raise ValueError(
            f"agent_permissions.mode must be one of {sorted(VALID_PERMISSION_MODES)}"
        )

    always_allow = raw.get("always_allow", [])
    if not isinstance(always_allow, list) or any(
        not isinstance(name, str) or not name.strip() for name in always_allow
    ):
        raise ValueError("agent_permissions.always_allow must be a list of tool names")

    deduped = sorted({name.strip() for name in always_allow})
    return {"mode": mode, "always_allow": deduped}


def load_agent_permissions(user_id: int) -> AgentPermissions:
    """Read the user's stored permissions, falling back to defaults on any error.

    Falling back to the default mode keeps chat working when settings are
    missing or malformed; the default posture still requires approval for
    side-effecting tools, so the failure mode is never an silent auto-approve.
    """
    try:
        from app.models.database.user_settings import get_user_settings

        settings = get_user_settings(user_id)
        raw = settings.agent_permissions if settings else {}
        normalized = normalize_agent_permissions(raw or {})
        return AgentPermissions(
            mode=normalized["mode"],
            always_allow=frozenset(normalized["always_allow"]),
        )
    except Exception:
        logger.warning(
            "Could not load agent permissions for user %s; using defaults",
            user_id,
            exc_info=True,
        )
        return AgentPermissions()
