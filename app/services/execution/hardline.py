"""Unconditional blocklist for commands with no recovery path.

Minimal port of Hermes-agent's "hardline floor": these patterns are refused
on host-reaching backends even when the user's permission mode is
auto_approve, because approving an agent to work with your files is not the
same as approving it to wipe the disk or power the machine off. Sandboxed
backends skip this check — inside an isolated container these commands are
harmless.
"""

from __future__ import annotations

import re


_HARDLINE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), description)
    for pattern, description in (
        (
            r"(?:^|[;&|]\s*|\bsudo\s+)rm\s+(?:-[a-z]*\s+)*(?:-[a-z]*[rf][a-z]*\s+)+(?:--\S+\s+)*(?:/|/\*)\s*(?:$|[;&|])",
            "recursive delete of filesystem root",
        ),
        (r"(?:^|[;&|`$(\s])mkfs(?:\.\w+)?\s", "filesystem format"),
        (
            r"(?:^|[;&|`$(\s])dd\s+[^;&|]*of=/dev/(?:sd|hd|nvme|disk|mmcblk)",
            "raw write to block device",
        ),
        (r"(?:^|[;&|`$(\s])(?:shutdown|reboot|poweroff|halt)(?:\s|$)", "system shutdown/reboot"),
        (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
        (r"(?:^|[;&|`$(\s])kill\s+(?:-9\s+)?-1(?:\s|$)", "kill all processes"),
        (
            r"(?:^|[;&|`$(\s])chmod\s+(?:-[a-z]+\s+)*(?:777|000)\s+/\s*(?:$|[;&|])",
            "permission change on filesystem root",
        ),
        (r">\s*/dev/(?:sd|hd|nvme|mmcblk)", "redirect onto block device"),
        (
            r"(?:^|[;&|]\s*)rm\s+(?:-[a-z]*\s+)*(?:-[a-z]*[rf][a-z]*\s+)+(?:--\S+\s+)*(?:\.|\./|\*|/workspace/?|/workspace/\*)\s*(?:$|[;&|])",
            "recursive delete of workspace root",
        ),
        (r"(?:^|[;&|]\s*)git\s+reset\s+--hard(?:\s|$)", "discarding Git reset"),
        (r"(?:^|[;&|]\s*)git\s+clean\s+-[a-z]*f[a-z]*(?:\s|$)", "forced Git clean"),
        (
            r"(?:^|[;&|]\s*)git\s+(?:checkout\s+--|restore)\s+\.(?:\s|$)",
            "discarding all workspace changes",
        ),
    )
)


def detect_hardline_command(command: str) -> str | None:
    """Return a description when the command matches the hardline floor."""
    normalized = " ".join((command or "").split())
    for pattern, description in _HARDLINE_PATTERNS:
        if pattern.search(normalized):
            return description
    return None
