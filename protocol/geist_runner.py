"""Bridge from the compute protocol to Geist's agent inference stack."""

from protocol.models import InferenceRecord, InferenceRequest


class GeistAgentRunner:
    """Adapts any Geist ``BaseAgent`` to the protocol's ``InferenceRunner``.

    Requests should use temperature 0 so re-execution is reproducible; sampled
    decoding is only auditable when the backing runner honors seeds
    deterministically (see docs/compute_protocol.md, "Determinism").
    """

    def __init__(self, agent):
        self._agent = agent

    def run(self, model_id: str, request: InferenceRequest) -> InferenceRecord:
        completions = self._agent.complete_text(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            n=1,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        output = completions[0] if isinstance(completions, list) else str(completions)
        return InferenceRecord(request_id=request.request_id, output_text=output)
