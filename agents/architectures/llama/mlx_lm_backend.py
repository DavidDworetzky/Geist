"""Adapter from Geist's completion contract to the optional mlx-lm runtime."""

import importlib
import time
from collections.abc import Iterator

from agents.models.llama_completion import strings_to_message_dict


def _configure_thread_local_generation_stream() -> None:
    mx = importlib.import_module("mlx.core")
    generate_module = importlib.import_module("mlx_lm.generate")
    generate_module.generation_stream = mx.new_thread_local_stream(mx.default_device())


class MLXLMBackend:
    """Load and generate with mlx-lm while matching ``LlamaMLX.complete``."""

    def __init__(
        self,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 1.0,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        weights_dir: str | None = None,
    ):
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "The mlx_lm implementation requires the pinned mlx-lm dependency."
            ) from exc

        _configure_thread_local_generation_stream()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_id = model_id
        self.weights_dir = weights_dir
        self.model, self.tokenizer = load(weights_dir or model_id)
        self.last_stats: dict[str, float] = {}

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self._build_messages_prompt(messages)

    def _build_messages_prompt(self, messages: list[dict[str, str | None]]) -> str:
        normalized = [
            {"role": message["role"], "content": message.get("content") or ""}
            for message in messages
        ]
        template_options = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "qwen3" in self.model_id.casefold():
            template_options["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(
            normalized,
            **template_options,
        )

    def stream_text(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield decoded text segments and retain mlx-lm timing statistics."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        yield from self.stream_messages(messages)

    def stream_messages(
        self,
        messages: list[dict[str, str | None]],
    ) -> Iterator[str]:
        """Yield decoded text for a structured conversation."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._build_messages_prompt(messages)
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        started = time.perf_counter()
        final_response = None
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
        ):
            final_response = response
            if response.text:
                yield response.text

        elapsed = time.perf_counter() - started
        if final_response is not None:
            self.last_stats = {
                "prompt_tokens": int(final_response.prompt_tokens),
                "prompt_tps": float(final_response.prompt_tps),
                "generation_tokens": int(final_response.generation_tokens),
                "generation_tps": float(final_response.generation_tps),
                "peak_memory_gb": float(final_response.peak_memory),
                "elapsed_seconds": elapsed,
            }

    def complete(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.complete_messages(messages)

    def complete_messages(
        self,
        messages: list[dict[str, str | None]],
    ) -> list[dict[str, str]]:
        response = "".join(self.stream_messages(messages)).strip()
        user_prompt = next(
            (
                message.get("content") or ""
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        return strings_to_message_dict(user_prompt, response)
