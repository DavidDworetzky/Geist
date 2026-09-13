"""Safe, actionable memory failures from local inference."""

import errno
from datetime import UTC, datetime
from typing import Literal


ModelLoadErrorCode = Literal["gpu_memory", "system_memory", "unified_memory"]


class ModelMemoryError(RuntimeError):
    def __init__(
        self,
        code: ModelLoadErrorCode,
        *,
        runtime: Literal["llama_server", "mlx_llama"],
        model_id: str | None = None,
        allow_system_ram: bool = False,
        offload_setting_available: bool = True,
        during_generation: bool = False,
    ) -> None:
        self.code = code
        self.model_id = model_id
        self.can_offload_to_system_ram = (
            runtime == "llama_server"
            and code == "gpu_memory"
            and not allow_system_ram
            and offload_setting_available
        )
        operation = "continue this response" if during_generation else "load this model"
        if code == "gpu_memory":
            detail = f"There is not enough available GPU memory to {operation}. "
            detail += (
                "Enable Allow system RAM in Settings > Models, or choose a smaller model."
                if self.can_offload_to_system_ram
                else "Close other GPU applications or choose a smaller model."
            )
        elif code == "unified_memory":
            detail = (
                f"There is not enough available shared memory to {operation}. "
                "Your Mac shares memory between its CPU and GPU. "
                "Close other applications or choose a smaller model."
            )
        else:
            detail = (
                f"There is not enough available system memory to {operation}. "
                "Close other applications or choose a smaller model."
            )
        if during_generation:
            detail += " A shorter conversation can also reduce memory use."
        super().__init__(detail)

    def to_status(self, model_id: str | None = None) -> dict:
        return {
            "model_id": self.model_id or model_id or "",
            "state": "failed",
            "detail": str(self),
            "error_code": self.code,
            "can_offload_to_system_ram": self.can_offload_to_system_ram,
            "started_at": None,
            "updated_at": datetime.now(UTC).isoformat(),
        }


def is_mlx_memory_error(error: Exception) -> bool:
    if isinstance(error, MemoryError):
        return True
    if isinstance(error, OSError) and error.errno == errno.ENOMEM:
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "outofmemory",
            "insufficient memory",
            "std::bad_alloc",
            "[malloc] unable to allocate",
            "[metal::malloc] unable to allocate",
        )
    )
