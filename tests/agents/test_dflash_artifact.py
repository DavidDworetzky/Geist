from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.architectures.llama.dflash_artifact import find_dflash_path
from agents.architectures.llama.mlx_lm_backend import MLXLMBackend


def test_missing_automatic_drafter_does_not_download(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_MLX_DFLASH", "auto")
    monkeypatch.delenv("GEIST_MLX_DFLASH_DIR", raising=False)
    monkeypatch.setattr(
        "agents.architectures.llama.dflash_artifact.default_dflash_path", lambda: tmp_path
    )
    assert find_dflash_path("Qwen/Qwen3.8-27B") is None
    assert list(tmp_path.iterdir()) == []


def test_explicit_missing_drafter_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_MLX_DFLASH", "on")
    monkeypatch.setenv("GEIST_MLX_DFLASH_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="download_mlx_dflash"):
        find_dflash_path("Qwen/Qwen3.8-27B")


def test_auto_discovery_requires_complete_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_MLX_DFLASH", "auto")
    monkeypatch.delenv("GEIST_MLX_DFLASH_DIR", raising=False)
    monkeypatch.setattr(
        "agents.architectures.llama.dflash_artifact.default_dflash_path", lambda: tmp_path
    )
    (tmp_path / "config.json").write_text("{}")
    assert find_dflash_path("Qwen/Qwen3.8-27B") is None
    (tmp_path / "model.safetensors").touch()
    assert find_dflash_path("qwen/qwen3.8-27b") == tmp_path
    assert find_dflash_path("Qwen/Qwen3.5-27B") is None
    monkeypatch.setenv("GEIST_MLX_DFLASH", "off")
    assert find_dflash_path("Qwen/Qwen3.8-27B") is None


def test_dflash_stream_filters_eos_finalizes_and_reports_stats():
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.model_id = "Qwen/Qwen3.8-27B"

    class Detokenizer:
        def reset(self):
            self.segment = ""
            self.tokens = []

        def add_token(self, token):
            self.segment = str(token)
            self.tokens.append(token)

        def finalize(self):
            self.segment = "!"

        @property
        def last_segment(self):
            result, self.segment = self.segment, ""
            return result

    detokenizer = Detokenizer()
    backend.tokenizer = SimpleNamespace(detokenizer=detokenizer, eos_token_ids={99})
    backend.max_new_tokens, backend.temperature, backend.top_p = 32, 0.7, 0.9
    backend._dflash = SimpleNamespace(last_stats={"generation_tps": 30})
    closed = []

    def generate(ids, **kwargs):
        assert ids == [1, 2]
        assert kwargs == {"max_tokens": 32, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
        try:
            yield from [3, 4, 99]
        finally:
            closed.append(True)

    backend._dflash.generate = generate
    assert "".join(backend._stream_dflash([1, 2])) == "34!"
    assert closed == [True]
    assert backend.last_stats == {"implementation": "mlx_dflash", "generation_tps": 30}
    assert detokenizer.tokens == [3, 4]


def test_dflash_loading_is_lazy_and_idempotent(monkeypatch):
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.model, backend.tokenizer = object(), object()
    backend.prefill_step_size = 2048
    import sys

    drafter = MagicMock()
    decoder = MagicMock()
    load = MagicMock(return_value=drafter)
    wrappers = [MagicMock(), MagicMock()]
    install = MagicMock(side_effect=[[wrappers[0]], [wrappers[1]]])
    tune = MagicMock(return_value=[{"split_k": 2}])
    monkeypatch.setattr(
        "agents.architectures.llama.dflash_artifact.find_dflash_path", lambda _: Path("/drafter")
    )
    monkeypatch.setitem(
        sys.modules,
        "agents.architectures.llama.dflash_backend",
        SimpleNamespace(load_drafter=load, DFlashDecoder=MagicMock(return_value=decoder)),
    )
    monkeypatch.setitem(
        sys.modules,
        "agents.architectures.llama.qwen_small_m",
        SimpleNamespace(install_small_m=install, tune_small_m=tune),
    )
    backend._prepare_dflash()
    backend._prepare_dflash()
    load.assert_called_once_with("/drafter", backend.model)
    tune.assert_called_once_with(wrappers)
    assert backend._dflash is decoder


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_initialization_failure_disables_wrappers_before_fallback(monkeypatch, mode):
    import sys

    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.model, backend.tokenizer = object(), object()
    backend.prefill_step_size = 2048
    backend._dflash = None
    wrapper = SimpleNamespace(enabled=True)
    monkeypatch.setenv("GEIST_MLX_DFLASH", mode)
    monkeypatch.setattr(
        "agents.architectures.llama.dflash_artifact.find_dflash_path", lambda _: Path("/drafter")
    )
    monkeypatch.setitem(
        sys.modules,
        "agents.architectures.llama.dflash_backend",
        SimpleNamespace(
            load_drafter=MagicMock(return_value=MagicMock()),
            DFlashDecoder=MagicMock(return_value=object()),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agents.architectures.llama.qwen_small_m",
        SimpleNamespace(
            install_small_m=MagicMock(side_effect=[[wrapper], []]),
            tune_small_m=MagicMock(side_effect=RuntimeError("unsupported kernel")),
        ),
    )
    if mode == "on":
        with pytest.raises(RuntimeError, match="unsupported kernel"):
            backend._prepare_dflash()
    else:
        backend._prepare_dflash()
        assert backend._dflash_checked
    assert backend._dflash is None
    assert not wrapper.enabled
