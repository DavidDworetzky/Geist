"""
Minimal Recursive Language Model (RLM) layer over BaseAgent.

An RlmAgent wraps any BaseAgent and runs a code-first loop: instead of the
JSON tool-call protocol, the model writes Python that executes in a
persistent REPL. Supplemental context lives in the REPL as a variable
(`context`), the wrapped model is callable as a function (`llm_query`), and
recursive delegation is a function call (`rlm_query`) that runs a child
RlmAgent with its own REPL. Adapter actions are reachable via `call_tool`.

Security: generated code runs via `exec` in this process — the same trust
level as adapter execution, not a sandbox. Swap RlmEnvironment for an
isolated implementation before exposing this to untrusted inputs.
"""

import contextlib
import io
import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import BaseAgent
from agents.tool_calling import ToolCall


logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 8
DEFAULT_MAX_DEPTH = 2
OUTPUT_CHAR_LIMIT = 4000
FALLBACK_CONTEXT_CHAR_LIMIT = 8000

RLM_SYSTEM_PROMPT = """You are a Recursive Language Model (RLM) agent. You solve the task by writing Python that runs in a persistent REPL; state carries over between your turns.

Available in the REPL:
- context: the task's supplemental context. It may be huge — inspect it with code (len(context), context[:2000], slicing, splitting); never assume you have seen all of it.
- llm_query(prompt): one-shot question to the underlying model, returns the answer text.
- rlm_query(prompt, context=None): delegate a subtask to a recursive child agent with its own REPL; returns its final answer text.
- call_tool("Adapter__action", **kwargs) and list_tools(): invoke registered tools.
- FINAL(answer): call this when done to end the session with your answer.

Rules:
- Respond with one ```python code block; you will see its printed output next turn. Print summaries, not whole variables — output is truncated.
- If the task needs no code, reply with the final answer as plain text and no code block."""

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class RlmResult:
    """Outcome of one RlmAgent session."""

    answer: str | None
    iterations: int
    transcript: list[dict[str, str]] = field(default_factory=list)


class RlmEnvironment:
    """Persistent in-process Python REPL backing one RlmAgent session."""

    def __init__(self, context: Any, bindings: dict[str, Any]):
        self.finished = False
        self.final_answer: str | None = None
        self.namespace: dict[str, Any] = {"context": context, "FINAL": self._final, **bindings}

    def _final(self, answer: Any) -> None:
        self.finished = True
        self.final_answer = answer if isinstance(answer, str) else str(answer)

    def execute(self, code: str) -> str:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, self.namespace)  # noqa: S102 - the RLM execution surface
        except Exception:
            buffer.write(traceback.format_exc(limit=8))
        output = buffer.getvalue()
        return output if output.strip() else "(no output)"


class RlmAgent:
    """
    Code-first agent loop over a wrapped BaseAgent.

    Works with any provider the wrapped agent supports; recursion reuses the
    same underlying agent at increasing depth with a fresh REPL per child.
    """

    def __init__(
        self,
        agent: BaseAgent,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_tokens: int | None = None,
        depth: int = 0,
    ):
        self._agent = agent
        self._max_iterations = max_iterations
        self._max_depth = max_depth
        self._max_tokens = max_tokens
        self._depth = depth

    def run(self, prompt: str, context: Any = None) -> RlmResult:
        environment = RlmEnvironment(
            context,
            bindings={
                "llm_query": self._llm_query,
                "rlm_query": self._rlm_query,
                "call_tool": self._call_tool,
                "list_tools": self._list_tools,
            },
        )
        transcript: list[dict[str, str]] = []

        for iteration in range(1, self._max_iterations + 1):
            model_text = self._complete(self._render_prompt(prompt, context, transcript))
            if not model_text.strip():
                transcript.append({"code": "", "output": "(empty completion)"})
                continue

            code = self._extract_code(model_text)
            if code is None:
                return RlmResult(answer=model_text.strip(), iterations=iteration, transcript=transcript)

            output = _truncate_middle(environment.execute(code), OUTPUT_CHAR_LIMIT)
            transcript.append({"code": code, "output": output})
            logger.debug("RLM depth=%d iteration=%d output: %s", self._depth, iteration, output)

            if environment.finished:
                return RlmResult(
                    answer=environment.final_answer, iterations=iteration, transcript=transcript
                )

        logger.warning("RLM session exhausted %d iterations without FINAL", self._max_iterations)
        return RlmResult(answer=None, iterations=self._max_iterations, transcript=transcript)

    # ------------------------------------------------------------------
    # REPL bindings
    # ------------------------------------------------------------------

    def _llm_query(self, prompt: str, system_prompt: str | None = None) -> str:
        completion = self._agent.complete_text(
            prompt=prompt, system_prompt=system_prompt, max_tokens=self._max_tokens
        )
        texts = self._agent._transform_completions(completion)
        return texts[0] if texts else ""

    def _rlm_query(self, prompt: str, context: Any = None) -> str:
        if self._depth + 1 >= self._max_depth:
            flattened = prompt
            if context is not None:
                flattened += "\n\nCONTEXT:\n" + _truncate_middle(
                    str(context), FALLBACK_CONTEXT_CHAR_LIMIT
                )
            return self._llm_query(flattened)
        child = RlmAgent(
            self._agent,
            max_iterations=self._max_iterations,
            max_depth=self._max_depth,
            max_tokens=self._max_tokens,
            depth=self._depth + 1,
        )
        result = child.run(prompt, context=context)
        return result.answer if result.answer is not None else ""

    def _call_tool(self, name: str, **kwargs: Any) -> Any:
        call = ToolCall.from_qualified_name(name, kwargs)
        result = self._agent._agent_context.get_tool_dispatcher().dispatch(call)
        if not result.success:
            raise RuntimeError(f"Tool {name} failed: {result.error}")
        return result.result

    def _list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": schema.qualified_name,
                "description": schema.description,
                "parameters": schema.parameters.get("properties", {}),
            }
            for schema in self._agent._agent_context.get_tool_schemas()
        ]

    # ------------------------------------------------------------------
    # Loop internals
    # ------------------------------------------------------------------

    def _complete(self, prompt: str) -> str:
        completion = self._agent.complete_text(
            prompt=prompt, system_prompt=RLM_SYSTEM_PROMPT, max_tokens=self._max_tokens
        )
        texts = self._agent._transform_completions(completion)
        return texts[0] if texts and texts[0] else ""

    @staticmethod
    def _extract_code(model_text: str) -> str | None:
        blocks = _CODE_FENCE_RE.findall(model_text)
        if not blocks:
            return None
        return "\n\n".join(block.strip() for block in blocks)

    def _render_prompt(self, prompt: str, context: Any, transcript: list[dict[str, str]]) -> str:
        parts = [f"TASK:\n{prompt}"]
        if context is not None:
            parts.append(
                f"A `context` variable is loaded in the REPL "
                f"(type {type(context).__name__}, str length {len(str(context))})."
            )
        for index, entry in enumerate(transcript, start=1):
            parts.append(
                f"--- code (iteration {index}) ---\n{entry['code']}\n"
                f"--- output ---\n{entry['output']}"
            )
        parts.append(
            "Write the next ```python block (call FINAL(answer) when done), "
            "or reply with plain text to answer directly."
        )
        return "\n\n".join(parts)


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} chars truncated] ...\n{text[-half:]}"
