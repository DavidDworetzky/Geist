"""
Voice chat WebSocket endpoint for real-time audio streaming.
"""
import json
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from app.services.voice_session import VoiceSessionService
from agents.base_agent import BaseAgent
from agents.agent_type import AgentType
from app.services.user_settings_service import UserSettingsService
from agents.agent_context import AgentContext
from app.models.user_settings import AgentConfigRequest
from agents.prompt.prompt import AGENT_PROMPTS

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


@router.websocket("/stream")
async def voice_stream_websocket(
    websocket: WebSocket,
    session_id: int = Query(..., description="Chat session ID"),
    agent_type: str = Query("online", description="Agent type (online or local)"),
    stt_provider: str = Query("mms", description="STT provider (mms or whisper)"),
    tts_provider: str = Query("sesame", description="TTS provider (sesame or openai)")
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
    await websocket.accept()
    logger.info(f"Voice WebSocket connected: session_id={session_id}, agent_type={agent_type}")
    
    voice_service: Optional[VoiceSessionService] = None
    
    try:
        # Get agent context (you may need to pass this properly)
        from app.main import get_default_agent_context
        agent_context = get_default_agent_context()
        
        # Create agent
        agent = await get_agent_for_session(agent_type, agent_context)
        
        # Create voice service
        import os
        provider_kwargs = {}
        
        if stt_provider == "whisper":
            provider_kwargs["whisper_api_key"] = os.getenv("OPENAI_API_KEY")
        
        if tts_provider == "openai":
            provider_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        elif tts_provider == "sesame":
            import torch
            provider_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        voice_service = VoiceSessionService(
            agent=agent,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            **provider_kwargs
        )
        
        # Send ready message
        await websocket.send_json({"type": "ready"})
        
        while True:
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
        except:
            pass
    finally:
        if voice_service:
            voice_service.reset()
        try:
            await websocket.close()
        except:
            pass


@router.post("/upload")
async def voice_upload(
    audio_file: bytes,
    session_id: int = Query(...),
    agent_type: str = Query("online"),
    stt_provider: str = Query("mms"),
    tts_provider: str = Query("sesame")
):
    """
    HTTP fallback endpoint for voice interaction.
    
    Upload an audio clip, get back transcript, text response, and audio response.
    """
    try:
        # Get agent context and create agent
        from app.main import get_default_agent_context
        agent_context = get_default_agent_context()
        agent = await get_agent_for_session(agent_type, agent_context)
        
        # Setup provider kwargs
        import os
        provider_kwargs = {}
        if stt_provider == "whisper":
            provider_kwargs["whisper_api_key"] = os.getenv("OPENAI_API_KEY")
        if tts_provider == "openai":
            provider_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        elif tts_provider == "sesame":
            import torch
            provider_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create voice service
        voice_service = VoiceSessionService(
            agent=agent,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            **provider_kwargs
        )
        
        # Add audio and get transcript
        import numpy as np
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
        import base64
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

