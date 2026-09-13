"""A gated model source with the real local-agent/MLX adapter stack above it."""

from collections.abc import Iterator
from threading import Event, Lock
from typing import Any

from agents.architectures.chat_template_tools import provider_tool_name
from agents.architectures.llama.mlx_lm_backend import MLXLMBackend
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.local_agent import LocalAgent


class StreamingProbe:
    def __init__(self) -> None:
        # Process-global probe: Playwright must keep workers=1 / fullyParallel=false.
        self._lock = Lock()
        self._generation = 0
        self.active = False
        self.stage = 0
        self.closed = False
        self.tools_seen = False
        self.scenario = "text"
        self.search_calls: list[dict[str, Any]] = []
        self.tool_result_seen = False
        self.gates = [Event(), Event()]

    def reset(self) -> None:
        with self._lock:
            self._reset()

    def _reset(self) -> None:
        self._generation += 1
        self.active = False
        self.stage = 0
        self.closed = False
        self.tools_seen = False
        self.search_calls = []
        self.tool_result_seen = False
        for gate in self.gates:
            gate.set()

    def start(self, scenario: str = "text") -> None:
        with self._lock:
            self._reset()
            self.gates = [Event(), Event()]
            self.scenario = scenario
            self.active = True

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self.active,
                "stage": self.stage,
                "closed": self.closed,
                "tools_seen": self.tools_seen,
                "released": [gate.is_set() for gate in self.gates],
                "search_calls": list(self.search_calls),
                "tool_result_seen": self.tool_result_seen,
            }

    def segments(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> Iterator[str]:
        if not tools:
            raise AssertionError("The streaming regression must exercise the tool-enabled path")
        with self._lock:
            self.tools_seen = True
            gates, generation = self.gates, self._generation
            scenario = self.scenario
        if scenario == "xml_tool":
            complete_name = provider_tool_name("agent.goal.complete")
            if self.closed and any(tool["function"]["name"] == complete_name for tool in tools):
                import json

                observation = next(
                    item["tool_call_id"] for item in reversed(messages) if item["role"] == "tool"
                )
                yield (
                    f"<tool_call>\n<function={complete_name}>\n"
                    "<parameter=summary>STREAM-FIRST STREAM-SECOND STREAM-FINAL</parameter>\n"
                    '<parameter=evidence>["Returned the fixture search result"]</parameter>\n'
                    f"<parameter=evidence_refs>{json.dumps([observation])}</parameter>\n"
                    "</function>\n</tool_call>"
                )
                return
            if messages[-1]["role"] != "tool":
                name = provider_tool_name("web.search")
                if not any(tool["function"]["name"] == name for tool in tools or []):
                    raise AssertionError("Expected the search tool to be offered")
                call = (
                    f"<tool_call>\n<function={name}>\n"
                    "<parameter=query>\nrecent celebrity headlines\n</parameter>\n"
                    "<parameter=max_results>\n3\n</parameter>\n"
                    "</function>\n</tool_call>"
                )
                for index in range(0, len(call), 7):
                    with self._lock:
                        if generation != self._generation:
                            return
                    yield call[index : index + 7]
                return
            with self._lock:
                if generation != self._generation:
                    return
                self.tool_result_seen = "Fixture celebrity headline" in messages[-1]["content"]
                if not self.tool_result_seen:
                    raise AssertionError("Expected the search result in model history")
        try:
            for index, segment in enumerate(("STREAM-FIRST", " STREAM-SECOND", " STREAM-FINAL")):
                with self._lock:
                    if generation != self._generation:
                        return
                    self.stage = index + 1
                yield segment
                # Both SSE iteration and synchronous control routes use workers;
                # never put this blocking wait on an async event-loop thread.
                if index < len(gates) and not gates[index].wait(timeout=10):
                    raise TimeoutError(
                        "Browser did not receive a chunk before releasing generation"
                    )
        finally:
            with self._lock:
                if generation == self._generation:
                    self.closed = True

    def agent(self) -> LocalAgent:
        backend = ProbeMLXBackend(self)
        runner = MLXLlamaRunner()
        runner.llama = backend
        runner.model_id = backend.model_id
        runner.implementation = "mlx_lm"
        runner.supports_native_tool_calling = True
        # Bypass weight loading only; retain production agent/runner/adapter dispatch.
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
        yield from self.probe.segments(messages, tools)
