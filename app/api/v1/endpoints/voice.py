"""
Voice chat WebSocket endpoint for real-time audio streaming.
"""

import asyncio
import base64
import json
import logging
import os
from contextlib import aclosing, suppress
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from agents.agent_context import AgentContext
from agents.base_agent import BaseAgent
from agents.prompt.prompt import AGENT_PROMPTS
from app.models.user_settings import AgentConfigRequest
from app.services.agent_context_provider import get_default_agent_context
from app.services.user_settings_service import UserSettingsService


logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_PROMPT = AGENT_PROMPTS.get("default", "You are a helpful assistant.")


async def get_agent_for_session(
    agent_type: str,
    agent_context: AgentContext,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> BaseAgent:
    """
    Get or create an agent for the voice session.

    Args:
        agent_type: Type of agent ("online" or "local")
        agent_context: Agent context
        max_tokens: Max tokens override
        temperature: Temperature override
        top_p: Top-p override

    Returns:
        BaseAgent: Configured agent
    """
    # Create config overrides
    overrides = AgentConfigRequest(
        agent_type=agent_type.lower() if agent_type else None,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Create the agent from workspace-owned settings.
    agent = await asyncio.to_thread(
        UserSettingsService.create_agent_from_default_workspace, agent_context, overrides
    )
    return agent


def _build_provider_kwargs(
    stt_provider: str,
    tts_provider: str,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_language: str | None = None,
    tts_instruct: str | None = None,
    tts_speed: float | None = None,
) -> dict[str, Any]:
    """
    Build provider-specific keyword arguments for STT and TTS providers.

    Args:
        stt_provider: Selected STT provider name.
        tts_provider: Selected TTS provider name.
        tts_model: Optional TTS model override.
        tts_voice: Optional TTS voice override.
        tts_language: Optional language code override.
        tts_instruct: Optional style/instruction prompt for capable providers.
        tts_speed: Optional speech speed multiplier.

    Returns:
        Dict[str, Any]: Keyword arguments to pass into `VoiceSessionService`.
    """
    provider_kwargs: dict[str, Any] = {}

    if stt_provider == "whisper":
        provider_kwargs["whisper_api_key"] = os.getenv("OPENAI_API_KEY")

    if tts_provider == "openai":
        provider_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    if tts_model:
        provider_kwargs["model"] = tts_model
    if tts_voice:
        provider_kwargs["voice"] = tts_voice
    if tts_language:
        provider_kwargs["language"] = tts_language
    if tts_instruct:
        provider_kwargs["instruct"] = tts_instruct
    if tts_speed is not None:
        provider_kwargs["speed"] = tts_speed

    return provider_kwargs


@router.get("/models")
async def list_voice_models():
    """Return supported voice/TTS providers and model options for frontend selection."""
    from app.services.tts import get_supported_tts_providers

    providers = get_supported_tts_providers()
    local_ready = next(
        (
            provider["provider"]
            for provider in providers
            if provider.get("type") == "local"
            and any(
                model.get("artifact", {}).get("status") == "installed"
                and model.get("artifact", {}).get("runtime_ready") is True
                for model in provider.get("models", [])
            )
        ),
        None,
    )
    return {
        "default_provider": local_ready or "sesame",
        "providers": providers,
    }


@router.websocket("/stream")
async def voice_stream_websocket(
    websocket: WebSocket,
    session_id: int = Query(..., description="Chat session ID"),
    agent_type: str = Query("online", description="Agent type (online or local)"),
    stt_provider: str = Query("mms", description="STT provider (mms or whisper)"),
    tts_provider: str = Query(
        "sesame",
        description="TTS provider (sesame, openai, kokoro, qwen3, or magpie)",
    ),
    tts_model: str | None = Query(None, description="TTS model ID override"),
    tts_voice: str | None = Query(None, description="TTS voice override"),
    tts_language: str | None = Query(None, description="TTS language code override"),
    tts_instruct: str | None = Query(None, description="TTS style/instruction prompt"),
    tts_speed: float | None = Query(None, description="TTS speed multiplier"),
):
    """
    WebSocket endpoint for real-time voice streaming.

    Protocol:
    - Client sends binary audio frames (PCM 16-bit signed int, mono, 16kHz)
    - Server sends JSON messages:
      - {"type": "transcript_partial", "text": "..."}
      - {"type": "transcript_final", "text": "..."}
      - {"type": "text_start"}
      - {"type": "text_chunk", "text": "..."}
      - {"type": "text_complete", "text": "..."}
      - {"type": "audio_start", "encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1}
      - {"type": "audio_chunk"} followed by binary audio data
      - {"type": "audio_complete"}
      - {"type": "done"}
      - {"type": "error", "message": "..."}
    """
    from app.services.voice_session import VoiceSessionService

    logger.info(
        "Voice WebSocket connection attempt: "
        f"session_id={session_id}, agent_type={agent_type}, stt={stt_provider}, "
        f"tts={tts_provider}, tts_model={tts_model}, tts_voice={tts_voice}"
    )
    await websocket.accept()
    logger.info(f"Voice WebSocket accepted: session_id={session_id}, agent_type={agent_type}")

    voice_service: VoiceSessionService | None = None
    stop_requested: bool = False
    response_task: asyncio.Task | None = None
    partial_task: asyncio.Task | None = None
    last_partial_samples = 0

    async def cancel_turn() -> None:
        nonlocal response_task, partial_task
        for task in (response_task, partial_task):
            if task is not None:
                task.cancel()
                try:
                    with suppress(asyncio.CancelledError):
                        await task
                except Exception:
                    logger.debug("Voice task ended after connection failure", exc_info=True)
        response_task = partial_task = None

    async def respond(audio) -> None:
        assert voice_service is not None
        try:
            transcript = await voice_service.transcribe(audio)
            if transcript.strip():
                await websocket.send_json({"type": "transcript_final", "text": transcript})
                async with aclosing(
                    voice_service.process_with_agent(
                        transcript=transcript, chat_id=session_id, system_prompt=DEFAULT_PROMPT
                    )
                ) as responses:
                    async for response in responses:
                        if response["type"] == "audio_chunk":
                            await websocket.send_bytes(response["audio"])
                        else:
                            await websocket.send_json(response)
            await websocket.send_json({"type": "done"})
        except Exception as error:
            await websocket.send_json({"type": "error", "message": str(error)})
            await websocket.send_json({"type": "done"})

    async def transcribe_partial(audio) -> None:
        assert voice_service is not None
        try:
            transcript = await voice_service.transcribe(audio)
            if transcript:
                await websocket.send_json({"type": "transcript_partial", "text": transcript})
        except Exception:
            logger.debug("Partial STT unavailable", exc_info=True)

    try:
        # Get agent context and create agent
        agent_context = get_default_agent_context()
        agent = await get_agent_for_session(agent_type=agent_type, agent_context=agent_context)

        # Create voice service with provider-specific kwargs
        provider_kwargs = _build_provider_kwargs(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_language=tts_language,
            tts_instruct=tts_instruct,
            tts_speed=tts_speed,
        )
        voice_service = VoiceSessionService(
            agent=agent, stt_provider=stt_provider, tts_provider=tts_provider, **provider_kwargs
        )

        # Send ready message
        await websocket.send_json({"type": "ready"})

        while websocket.client_state == WebSocketState.CONNECTED and not stop_requested:
            # Receive audio chunk
            try:
                data = await websocket.receive()
                if data["type"] == "websocket.disconnect":
                    break

                # Handle text messages (control)
                if "text" in data:
                    message = json.loads(data["text"])
                    if message.get("type") == "reset":
                        await cancel_turn()
                        voice_service.reset()
                        last_partial_samples = 0
                        await websocket.send_json({"type": "reset_complete"})
                        continue
                    if message.get("type") in {"stop", "close", "end"}:
                        stop_requested = True
                        await websocket.send_json({"type": "stopped"})
                        break

                # Handle binary audio
                if "bytes" in data:
                    # Half-duplex: do not accumulate the assistant's speech as
                    # another user turn while inference is active.
                    if response_task is not None and not response_task.done():
                        continue
                    audio_chunk = data["bytes"]
                    voice_service.add_audio_chunk(audio_chunk)
                    if voice_service.check_phrase_boundary():
                        await cancel_turn()
                        audio = voice_service.take_audio()
                        last_partial_samples = 0
                        await websocket.send_json({"type": "processing"})
                        response_task = asyncio.create_task(respond(audio))
                    elif voice_service.buffered_samples - last_partial_samples >= 16000 and (
                        partial_task is None or partial_task.done()
                    ):
                        import numpy as np

                        last_partial_samples = voice_service.buffered_samples
                        partial_task = asyncio.create_task(
                            transcribe_partial(np.concatenate(list(voice_service.audio_buffer)))
                        )

            except WebSocketDisconnect:
                logger.info(f"Voice WebSocket disconnected: session_id={session_id}")
                break
            except Exception as e:
                logger.error(f"Error processing voice data: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "message": str(e)})

    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception as send_error:
            logger.debug(f"Failed to send error message to WebSocket: {send_error}")
    finally:
        await cancel_turn()
        if voice_service:
            voice_service.close()
        try:
            await websocket.close()
        except Exception as close_error:
            logger.debug(f"Failed to close WebSocket: {close_error}")


@router.post("/upload")
async def voice_upload(
    audio_file: bytes,
    session_id: int = Query(...),
    agent_type: str = Query("online"),
    stt_provider: str = Query("mms"),
    tts_provider: str = Query("sesame"),
    tts_model: str | None = Query(None),
    tts_voice: str | None = Query(None),
    tts_language: str | None = Query(None),
    tts_instruct: str | None = Query(None),
    tts_speed: float | None = Query(None),
):
    """
    HTTP fallback endpoint for voice interaction.

    Upload an audio clip, get back transcript, text response, and audio response.
    """
    voice_service = None
    try:
        import numpy as np

        from app.services.voice_session import VoiceSessionService

        # Get agent context and create agent
        agent_context = get_default_agent_context()
        agent = await get_agent_for_session(agent_type, agent_context)

        # Setup provider kwargs
        provider_kwargs = _build_provider_kwargs(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_language=tts_language,
            tts_instruct=tts_instruct,
            tts_speed=tts_speed,
        )

        # Create voice service
        voice_service = VoiceSessionService(
            agent=agent, stt_provider=stt_provider, tts_provider=tts_provider, **provider_kwargs
        )

        # Add audio and get transcript
        audio_np = np.frombuffer(audio_file, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe
        transcript = await voice_service.transcribe(audio_np)

        # Get agent response (non-streaming for HTTP)
        full_text = ""
        audio_chunks = []

        async for response in voice_service.process_with_agent(
            transcript=transcript,
            chat_id=session_id,
            system_prompt=DEFAULT_PROMPT,
            use_streaming=False,
        ):
            if response["type"] == "text_complete":
                full_text = response["text"]
            elif response["type"] == "audio_chunk":
                audio_chunks.append(response["audio"])

        # Combine audio chunks
        combined_audio = b"".join(audio_chunks)
        audio_base64 = base64.b64encode(combined_audio).decode()

        return {
            "transcript": transcript,
            "response_text": full_text,
            "audio_base64": audio_base64,
            "sample_rate": voice_service.tts.sample_rate,
        }

    except Exception as e:
        logger.error(f"Voice upload error: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        if voice_service is not None:
            voice_service.close()
