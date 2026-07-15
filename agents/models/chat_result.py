import base64
import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass
class WorkArtifact:
    id: str
    kind: str
    mime_type: str
    sha256: str
    filename: str | None = None
    data_base64: str | None = None
    url: str | None = None

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        kind: str,
        mime_type: str,
        filename: str | None = None,
    ) -> "WorkArtifact":
        return cls(
            id=f"artifact_{uuid.uuid4().hex}",
            kind=kind,
            mime_type=mime_type,
            filename=filename,
            sha256=hashlib.sha256(data).hexdigest(),
            data_base64=base64.b64encode(data).decode("ascii"),
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        kind: str,
        mime_type: str,
        filename: str | None = None,
    ) -> "WorkArtifact":
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Artifact URLs must use HTTP(S)")
        return cls(
            id=f"artifact_{uuid.uuid4().hex}",
            kind=kind,
            mime_type=mime_type,
            filename=filename,
            sha256=hashlib.sha256(url.encode("utf-8")).hexdigest(),
            url=url,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallResult:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "succeeded"
    result_summary: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    error: str | None = None
    requires_approval: bool = False

    @classmethod
    def create(
        cls,
        *,
        id: str | None = None,
        name: str,
        arguments: dict[str, Any] | None = None,
        status: str = "succeeded",
        result_summary: str | None = None,
        artifact_ids: list[str] | None = None,
        error: str | None = None,
        requires_approval: bool = False,
    ) -> "ToolCallResult":
        return cls(
            id=id or f"toolcall_{uuid.uuid4().hex}",
            name=name,
            arguments=arguments or {},
            status=status,
            result_summary=result_summary,
            artifact_ids=artifact_ids or [],
            error=error,
            requires_approval=requires_approval,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
