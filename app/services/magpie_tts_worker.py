"""Isolated NeMo-Speech.cpp Magpie worker used by the Geist TTS provider."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

from app.services.local_tts_process import write_frame


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magpie_model", ctypes.c_char_p),
        ("codec_model", ctypes.c_char_p),
        ("tokenizer_model_dir", ctypes.c_char_p),
        ("text_normalizer_model_dir", ctypes.c_char_p),
    ]


class RuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("speaker", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("codec_threads", ctypes.c_int32),
        ("seed", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("chunk_frames", ctypes.c_int32),
        ("codec_queue_depth", ctypes.c_int32),
        ("codec_history_frames", ctypes.c_int32),
        ("codec_future_frames", ctypes.c_int32),
        ("window_ms", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("override_temperature", ctypes.c_bool),
        ("cfg_scale", ctypes.c_float),
        ("override_cfg_scale", ctypes.c_bool),
        ("use_cfg", ctypes.c_bool),
        ("use_local_transformer", ctypes.c_bool),
        ("use_kv_cache", ctypes.c_bool),
        ("use_stateful_codec", ctypes.c_bool),
        ("codec_cpu", ctypes.c_bool),
        ("flush_partial_chunk", ctypes.c_bool),
        ("verbose", ctypes.c_bool),
        ("lt_backend", ctypes.c_int32),
        ("sampling_backend", ctypes.c_int32),
        ("uma_mode", ctypes.c_int32),
        ("longform_mode", ctypes.c_int32),
        ("lt_fp32", ctypes.c_bool),
    ]


class SynthesizerConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model", ctypes.POINTER(ModelConfig)),
        ("runtime", ctypes.POINTER(RuntimeConfig)),
        ("default_language_code", ctypes.c_char_p),
        ("default_voice_name", ctypes.c_char_p),
    ]


class SynthesisOptions(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("request_id", ctypes.c_char_p),
        ("language_code", ctypes.c_char_p),
        ("speaker", ctypes.c_int32),
        ("seed", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("override_temperature", ctypes.c_bool),
        ("cfg_scale", ctypes.c_float),
        ("override_cfg_scale", ctypes.c_bool),
        ("voice_name", ctypes.c_char_p),
        ("output_sample_rate", ctypes.c_int32),
    ]


class SynthesisStats(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("sample_rate", ctypes.c_int32),
        ("generated_frames", ctypes.c_int32),
        ("chunks", ctypes.c_int32),
        ("e2e_chunks", ctypes.c_int32),
        ("samples_written", ctypes.c_uint64),
        ("tokenizer_ms", ctypes.c_double),
        ("encoder_ms", ctypes.c_double),
        ("audio_s", ctypes.c_double),
        ("elapsed_s", ctypes.c_double),
        ("rtf", ctypes.c_double),
        ("rtfx", ctypes.c_double),
        ("ttfa_ms", ctypes.c_double),
        ("icl_avg_ms", ctypes.c_double),
        ("icl_min_ms", ctypes.c_double),
        ("icl_max_ms", ctypes.c_double),
        ("decoder_audio_s", ctypes.c_double),
        ("decoder_elapsed_s", ctypes.c_double),
        ("decoder_rtfx", ctypes.c_double),
        ("decoder_ttft_ms", ctypes.c_double),
        ("decoder_itl_avg_ms", ctypes.c_double),
        ("decoder_itl_min_ms", ctypes.c_double),
        ("decoder_itl_max_ms", ctypes.c_double),
        ("decoder_itl_p95_ms", ctypes.c_double),
        ("decoder_itl_p99_ms", ctypes.c_double),
        ("codec_audio_s", ctypes.c_double),
        ("codec_elapsed_s", ctypes.c_double),
        ("codec_rtfx", ctypes.c_double),
        ("codec_ttfa_ms", ctypes.c_double),
        ("codec_icl_avg_ms", ctypes.c_double),
        ("codec_icl_min_ms", ctypes.c_double),
        ("codec_icl_max_ms", ctypes.c_double),
        ("codec_icl_p95_ms", ctypes.c_double),
        ("codec_icl_p99_ms", ctypes.c_double),
        ("e2e_ttfa_ms", ctypes.c_double),
        ("e2e_icl_avg_ms", ctypes.c_double),
        ("e2e_icl_min_ms", ctypes.c_double),
        ("e2e_icl_max_ms", ctypes.c_double),
        ("e2e_icl_p95_ms", ctypes.c_double),
        ("e2e_icl_p99_ms", ctypes.c_double),
        ("e2e_rtfx", ctypes.c_double),
    ]


PCMCallback = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_void_p,
)


def _bind(library: ctypes.CDLL) -> None:
    library.nemo_speech_tts_runtime_config_default.restype = RuntimeConfig
    library.nemo_speech_tts_synthesis_options_default.restype = SynthesisOptions
    library.nemo_speech_tts_synthesis_stats_default.restype = SynthesisStats
    library.nemo_speech_tts_create.argtypes = [
        ctypes.POINTER(SynthesizerConfig),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.nemo_speech_tts_create.restype = ctypes.c_int32
    library.nemo_speech_tts_destroy.argtypes = [ctypes.c_void_p]
    library.nemo_speech_tts_sample_rate.argtypes = [ctypes.c_void_p]
    library.nemo_speech_tts_sample_rate.restype = ctypes.c_int32
    library.nemo_speech_tts_synthesize_text.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(SynthesisOptions),
        ctypes.c_char_p,
        PCMCallback,
        ctypes.c_void_p,
        ctypes.POINTER(SynthesisStats),
    ]
    library.nemo_speech_tts_synthesize_text.restype = ctypes.c_int32
    library.nemo_speech_tts_last_error.restype = ctypes.c_char_p


def _last_error(library: ctypes.CDLL) -> str:
    value = library.nemo_speech_tts_last_error()
    return value.decode("utf-8", errors="replace") if value else "unknown NeMo TTS error"


def _bytes_path(path: str) -> bytes:
    resolved = Path(path).expanduser().resolve(strict=True)
    return str(resolved).encode("utf-8")


def _run(args: argparse.Namespace) -> None:
    library_value = Path(args.library).expanduser()
    library_path = (
        str(library_value.resolve(strict=True))
        if library_value.is_absolute() or len(library_value.parts) > 1
        else args.library
    )
    library = ctypes.CDLL(library_path)
    _bind(library)

    magpie_model = _bytes_path(args.magpie_model)
    codec_model = _bytes_path(args.codec_model)
    tokenizer_dir = _bytes_path(args.tokenizer_dir)
    language = args.language.encode("utf-8")
    voice = args.voice.encode("utf-8")

    model_config = ModelConfig(
        size=ctypes.sizeof(ModelConfig),
        magpie_model=magpie_model,
        codec_model=codec_model,
        tokenizer_model_dir=tokenizer_dir,
        text_normalizer_model_dir=None,
    )
    runtime_config = library.nemo_speech_tts_runtime_config_default()
    runtime_config.lt_backend = 2  # NEMO_SPEECH_TTS_BACKEND_CUDA
    runtime_config.sampling_backend = 2
    synth_config = SynthesizerConfig(
        size=ctypes.sizeof(SynthesizerConfig),
        model=ctypes.pointer(model_config),
        runtime=ctypes.pointer(runtime_config),
        default_language_code=language,
        default_voice_name=voice,
    )
    synthesizer = ctypes.c_void_p()
    status = library.nemo_speech_tts_create(
        ctypes.byref(synth_config),
        ctypes.byref(synthesizer),
    )
    if status != 0:
        raise RuntimeError(_last_error(library))

    output = sys.stdout.buffer
    sample_rate = int(library.nemo_speech_tts_sample_rate(synthesizer))
    write_frame(output, b"R", json.dumps({"sample_rate": sample_rate}).encode("utf-8"))

    @PCMCallback
    def on_pcm(pcm, n_bytes, _user_data):
        write_frame(output, b"A", ctypes.string_at(pcm, n_bytes))
        return True

    try:
        for line in sys.stdin.buffer:
            request = json.loads(line.decode("utf-8"))
            if request.get("action") == "shutdown":
                return
            text = str(request.get("text") or "").strip()
            if not text:
                raise ValueError("TTS text must not be empty")

            request_language = str(request.get("language") or args.language).encode("utf-8")
            request_voice = str(request.get("voice") or args.voice).encode("utf-8")
            options = library.nemo_speech_tts_synthesis_options_default()
            options.language_code = request_language
            options.voice_name = request_voice
            options.speaker = -1
            stats = library.nemo_speech_tts_synthesis_stats_default()
            status = library.nemo_speech_tts_synthesize_text(
                synthesizer,
                ctypes.byref(options),
                text.encode("utf-8"),
                on_pcm,
                None,
                ctypes.byref(stats),
            )
            if status != 0:
                raise RuntimeError(_last_error(library))
            done = {
                "sample_rate": stats.sample_rate or sample_rate,
                "ttfa_ms": stats.e2e_ttfa_ms or stats.ttfa_ms,
                "rtf": stats.rtf,
            }
            write_frame(output, b"D", json.dumps(done).encode("utf-8"))
    finally:
        library.nemo_speech_tts_destroy(synthesizer)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--magpie-model", required=True)
    parser.add_argument("--codec-model", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--voice", default="John")
    return parser


def main() -> None:
    try:
        _run(_parser().parse_args())
    except Exception as error:
        write_frame(
            sys.stdout.buffer,
            b"E",
            json.dumps({"message": str(error)}).encode("utf-8"),
        )


if __name__ == "__main__":
    main()
