"""
Voice chat WebSocket endpoint for real-time audio streaming.
"""
import asyncio
import base64
import json
import logging
import os
from typing import Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState
import numpy as np

from agents.agent_context import AgentContext
from agents.base_agent import BaseAgent
from agents.prompt.prompt import AGENT_PROMPTS
from app.models.user_settings import AgentConfigRequest
from app.services.user_settings_service import UserSettingsService
from app.services.voice_session import VoiceSessionService
from app.services.agent_context_provider import get_default_agent_context
from app.services.tts import get_supported_tts_providers

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_PROMPT = AGENT_PROMPTS.get("default", "You are a helpful assistant.")


async def get_agent_for_session(
    agent_type: str,
    agent_context: AgentContext,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None
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
        top_p=top_p
    )
    
    # Create agent from user settings
    agent = UserSettingsService.create_agent_from_default_user(agent_context, overrides)
    return agent


def _build_provider_kwargs(
    stt_provider: str,
    tts_provider: str,
    tts_model: Optional[str] = None,
    tts_voice: Optional[str] = None,
    tts_language: Optional[str] = None,
    tts_instruct: Optional[str] = None,
    tts_speed: Optional[float] = None,
) -> Dict[str, Any]:
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
    provider_kwargs: Dict[str, Any] = {}

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
    return {
        "default_provider": "sesame",
        "providers": get_supported_tts_providers(),
    }


@router.websocket("/stream")
async def voice_stream_websocket(
    websocket: WebSocket,
    session_id: int = Query(..., description="Chat session ID"),
    agent_type: str = Query("online", description="Agent type (online or local)"),
    stt_provider: str = Query("mms", description="STT provider (mms or whisper)"),
    tts_provider: str = Query("sesame", description="TTS provider (sesame, openai, or qwen3)"),
    tts_model: Optional[str] = Query(None, description="TTS model ID override"),
    tts_voice: Optional[str] = Query(None, description="TTS voice override"),
    tts_language: Optional[str] = Query(None, description="TTS language code override"),
    tts_instruct: Optional[str] = Query(None, description="TTS style/instruction prompt"),
    tts_speed: Optional[float] = Query(None, description="TTS speed multiplier")
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
      - {"type": "audio_chunk"} followed by binary audio data
      - {"type": "audio_complete"}
      - {"type": "done"}
      - {"type": "error", "message": "..."}
    """
    logger.info(
        "Voice WebSocket connection attempt: "
        f"session_id={session_id}, agent_type={agent_type}, stt={stt_provider}, "
        f"tts={tts_provider}, tts_model={tts_model}, tts_voice={tts_voice}"
    )
    await websocket.accept()
    logger.info(f"Voice WebSocket accepted: session_id={session_id}, agent_type={agent_type}")
    
    voice_service: Optional[VoiceSessionService] = None
    stop_requested: bool = False
    
    try:
        # Get agent context and create agent
        agent_context = get_default_agent_context()
        agent = await get_agent_for_session(
            agent_type=agent_type,
            agent_context=agent_context
        )

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
            agent=agent,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            **provider_kwargs
        )
        
        # Send ready message
        await websocket.send_json({"type": "ready"})
        
        while websocket.client_state == WebSocketState.CONNECTED and not stop_requested:
            # Receive audio chunk
            try:
                data = await websocket.receive()
                
                # Handle text messages (control)
                if "text" in data:
                    message = json.loads(data["text"])
                    if message.get("type") == "reset":
                        voice_service.reset()
                        await websocket.send_json({"type": "reset_complete"})
                        continue
                    if message.get("type") in {"stop", "close", "end"}:
                        stop_requested = True
                        await websocket.send_json({"type": "stopped"})
                        break
                
                # Handle binary audio
                if "bytes" in data:
                    audio_chunk = data["bytes"]
                    
                    # Add to buffer and get partial transcript
                    partial_transcript = voice_service.add_audio_chunk(audio_chunk)
                    
                    if partial_transcript:
                        await websocket.send_json({
                            "type": "transcript_partial",
                            "text": partial_transcript
                        })
                    
                    # Check for phrase boundary
                    if voice_service.check_phrase_boundary():
                        # Get final transcript
                        final_transcript = voice_service.get_final_transcript()
                        
                        if final_transcript:
                            await websocket.send_json({
                                "type": "transcript_final",
                                "text": final_transcript
                            })
                            
                            # Process with agent and stream response
                            async for response in voice_service.process_with_agent(
                                transcript=final_transcript,
                                chat_id=session_id,
                                system_prompt=DEFAULT_PROMPT,
                                use_streaming=True
                            ):
                                if response["type"] == "audio_chunk":
                                    # Send audio as binary
                                    await websocket.send_json({"type": "audio_chunk_start"})
                                    await websocket.send_bytes(response["audio"])
                                else:
                                    # Send other messages as JSON
                                    await websocket.send_json(response)
                            
                            # Signal turn complete
                            await websocket.send_json({"type": "done"})
            
            except WebSocketDisconnect:
                logger.info(f"Voice WebSocket disconnected: session_id={session_id}")
                break
            except Exception as e:
                logger.error(f"Error processing voice data: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception as send_error:
            logger.debug(f"Failed to send error message to WebSocket: {send_error}")
    finally:
        if voice_service:
            voice_service.reset()
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
    tts_model: Optional[str] = Query(None),
    tts_voice: Optional[str] = Query(None),
    tts_language: Optional[str] = Query(None),
    tts_instruct: Optional[str] = Query(None),
    tts_speed: Optional[float] = Query(None)
):
    """
    HTTP fallback endpoint for voice interaction.
    
    Upload an audio clip, get back transcript, text response, and audio response.
    """
    try:
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
            agent=agent,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            **provider_kwargs
        )
        
        # Add audio and get transcript
        audio_np = np.frombuffer(audio_file, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        transcript = voice_service.stt.transcribe(audio_np, language="en")
        
        # Get agent response (non-streaming for HTTP)
        response_parts = []
        full_text = ""
        audio_chunks = []
        
        async for response in voice_service.process_with_agent(
            transcript=transcript,
            chat_id=session_id,
            system_prompt=DEFAULT_PROMPT,
            use_streaming=False
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
            "sample_rate": voice_service.tts.sample_rate
        }
    
    except Exception as e:
        logger.error(f"Voice upload error: {e}", exc_info=True)
        return {
            "error": str(e)
        }
