"""Thread-safe lifecycle state for local model initialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Literal


ModelLoadState = Literal["unloaded", "loading", "ready", "failed"]


@dataclass(frozen=True)
class ModelLoadStatus:
    model_id: str
    state: ModelLoadState
    detail: str
    started_at: datetime | None
    updated_at: datetime

    def to_dict(self) -> dict:
        return asdict(self)


class ModelLoadStatusRegistry:
    """Keep model load state aligned with process-local runner instances."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._statuses: dict[str, ModelLoadStatus] = {}

    def get(self, model_id: str) -> ModelLoadStatus:
        with self._lock:
            status = self._statuses.get(model_id)
            if status is not None:
                return status
        now = datetime.now(UTC)
        return ModelLoadStatus(
            model_id=model_id,
            state="unloaded",
            detail="Model is not loaded in this backend process.",
            started_at=None,
            updated_at=now,
        )

    def mark_loading(self, model_id: str, detail: str) -> ModelLoadStatus:
        with self._lock:
            current = self._statuses.get(model_id)
            now = datetime.now(UTC)
            status = ModelLoadStatus(
                model_id=model_id,
                state="loading",
                detail=detail,
                started_at=current.started_at if current and current.started_at else now,
                updated_at=now,
            )
            self._statuses[model_id] = status
            return status

    def mark_ready(self, model_id: str) -> ModelLoadStatus:
        return self._set_terminal(model_id, "ready", "Model is loaded and ready.")

    def mark_failed(self, model_id: str, detail: str) -> ModelLoadStatus:
        return self._set_terminal(model_id, "failed", detail)

    def _set_terminal(
        self,
        model_id: str,
        state: Literal["ready", "failed"],
        detail: str,
    ) -> ModelLoadStatus:
        with self._lock:
            current = self._statuses.get(model_id)
            now = datetime.now(UTC)
            status = ModelLoadStatus(
                model_id=model_id,
                state=state,
                detail=detail,
                started_at=current.started_at if current else now,
                updated_at=now,
            )
            self._statuses[model_id] = status
            return status

    def clear(self) -> None:
        with self._lock:
            self._statuses.clear()


model_load_status_registry = ModelLoadStatusRegistry()
