"""Conservative extractive summarization for durable chat memory."""

import re
from typing import Any


_SECRET_MARKERS = ("password", "api key", "token", "secret", "private key")
_FACT_PATTERNS = (
    re.compile(r"\bremember(?: that)?\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bmy name is\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bi (?:strongly |really )?prefer\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bi (?:work|live)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
)


def is_secret_like(text: str) -> bool:
    return any(marker in text.lower() for marker in _SECRET_MARKERS)


def _safe_summary_text(value: Any) -> str:
    text = str(value or "").strip()
    if is_secret_like(text):
        return "[sensitive content omitted from memory]"
    return text


def summarize_entries(previous_summary: str, entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if previous_summary:
        lines.append(previous_summary.strip())
    for entry in entries:
        user = _safe_summary_text(entry.get("user"))
        assistant = _safe_summary_text(entry.get("ai"))
        if user:
            lines.append(f"User: {user[:600]}")
        if assistant:
            lines.append(f"Geist: {assistant[:600]}")
    return "\n".join(lines)[-3000:]


def extract_durable_facts(entries: list[dict[str, Any]]) -> list[str]:
    facts: list[str] = []
    for entry in entries:
        user_text = str(entry.get("user") or "").strip()
        if not user_text or is_secret_like(user_text):
            continue
        for pattern in _FACT_PATTERNS:
            match = pattern.search(user_text)
            if not match:
                continue
            value = " ".join(match.group(1).split()).strip(" \"'")
            if 2 <= len(value) <= 300:
                fact = f"The user asked Geist to remember: {value}"
                if fact not in facts:
                    facts.append(fact)
            break
    return facts
