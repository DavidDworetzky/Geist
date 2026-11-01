"""
Process-wide cache for heavy local voice models.

Voice WebSocket sessions construct fresh STT/TTS providers per connection;
without sharing, every connection would reload multi-gigabyte weights.
Each cached entry carries a lock because the underlying runtimes (NeMo,
Kokoro, CosyVoice) are not documented as safe for concurrent inference.
Stateful models (e.g. Silero VAD's recurrent state) must not be cached here.
"""

import threading
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SharedModel:
    model: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


_models: dict[Hashable, SharedModel] = {}
_models_lock = threading.Lock()


def get_shared_model(key: Hashable, loader: Callable[[], Any]) -> SharedModel:
    """Return the cached model for `key`, loading it once with `loader` on first use."""
    with _models_lock:
        entry = _models.get(key)
        if entry is None:
            entry = SharedModel(model=loader())
            _models[key] = entry
        return entry


def clear_shared_models() -> None:
    """Drop every cached model (used by tests and explicit unloads)."""
    with _models_lock:
        _models.clear()
