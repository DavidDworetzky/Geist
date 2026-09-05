"""A gated model source with the real local-agent/MLX adapter stack above it."""

from collections.abc import Iterator
from threading import Event
from typing import Any

from agents.architectures.llama.mlx_lm_backend import MLXLMBackend
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.local_agent import LocalAgent


class StreamingProbe:
    def __init__(self) -> None:
        self.active = False
        self.stage = 0
        self.closed = False
        self.tools_seen = False
        self.gates = [Event(), Event()]

    def reset(self) -> None:
        self.active = False
        for gate in self.gates:
            gate.set()

    def start(self) -> None:
        self.reset()
        self.gates = [Event(), Event()]
        self.stage = 0
        self.closed = False
        self.tools_seen = False
        self.active = True

    def state(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "stage": self.stage,
            "closed": self.closed,
            "tools_seen": self.tools_seen,
            "released": [gate.is_set() for gate in self.gates],
        }

    def segments(self, tools: list[dict[str, Any]] | None) -> Iterator[str]:
        self.tools_seen = bool(tools)
        if not self.tools_seen:
            raise AssertionError("The streaming regression must exercise the tool-enabled path")
        gates = self.gates
        try:
            for index, segment in enumerate(("STREAM-FIRST", " STREAM-SECOND", " STREAM-FINAL")):
                self.stage = index + 1
                yield segment
                if index < len(gates) and not gates[index].wait(timeout=15):
                    raise TimeoutError(
                        "Browser did not receive a chunk before releasing generation"
                    )
        finally:
            self.closed = True

    def agent(self) -> LocalAgent:
        backend = ProbeMLXBackend(self)
        runner = MLXLlamaRunner()
        runner.llama = backend
        runner.model_id = backend.model_id
        runner.implementation = "mlx_lm"
        runner.supports_native_tool_calling = True
        agent = LocalAgent.__new__(LocalAgent)
        agent.runner = runner
        agent.runner_type = "mlx_llama"
        agent.model_id = backend.model_id
        agent.supports_native_tool_calling = True
        return agent


class ProbeMLXBackend(MLXLMBackend):
    def __init__(self, probe: StreamingProbe) -> None:
        self.probe = probe
        self.model_id = "Qwen/Qwen3.8-27B"
        self.supports_native_tool_calling = True

    def stream_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        del messages
        yield from self.probe.segments(tools)
