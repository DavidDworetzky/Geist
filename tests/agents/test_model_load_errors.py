import errno

import pytest

from agents.model_load_errors import ModelMemoryError, is_mlx_memory_error


@pytest.mark.parametrize(
    "error",
    [
        MemoryError(),
        OSError(errno.ENOMEM, "Cannot allocate memory"),
        RuntimeError("[METAL] Metal allocator out of memory"),
        RuntimeError(
            "[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        ),
        RuntimeError("[malloc] Unable to allocate 123456 bytes."),
        RuntimeError("[metal::malloc] Unable to allocate 123456 bytes."),
        RuntimeError("std::bad_alloc"),
    ],
)
def test_identifies_mlx_allocation_errors(error):
    assert is_mlx_memory_error(error)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("[METAL] Failed to compile shader"),
        RuntimeError("[metal::malloc] Resource limit (499000) exceeded."),
        RuntimeError("[metal::set_wired_limit] Invalid memory limit"),
        FileNotFoundError("/models/out of memory/config.json"),
        ValueError("Invalid model configuration"),
        OSError(errno.ENOSPC, "No space left on device"),
    ],
)
def test_does_not_label_unrelated_errors_as_memory_exhaustion(error):
    assert not is_mlx_memory_error(error)


@pytest.mark.parametrize(
    "runtime,code,allow,expected",
    [
        ("llama_server", "gpu_memory", False, True),
        ("llama_server", "gpu_memory", True, False),
        ("llama_server", "system_memory", False, False),
        ("mlx_llama", "unified_memory", False, False),
        ("mlx_llama", "unified_memory", True, False),
        ("mlx_llama", "gpu_memory", False, False),
    ],
)
def test_offload_guidance_requires_llama_and_disabled_offloading(runtime, code, allow, expected):
    error = ModelMemoryError(code, runtime=runtime, allow_system_ram=allow)
    assert error.can_offload_to_system_ram is expected
    assert ("Enable Allow system RAM" in str(error)) is expected


def test_unified_memory_generation_message_explains_recovery():
    error = ModelMemoryError("unified_memory", runtime="mlx_llama", during_generation=True)
    assert "continue this response" in str(error)
    assert "shares memory" in str(error)
    assert "shorter conversation" in str(error)
    assert "load this model" not in str(error)
