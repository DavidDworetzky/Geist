"""
Unit tests for voice session service.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from app.services.voice_session import VoiceSessionService


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.complete_text = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    ))
    agent.stream_complete_text = Mock(return_value=iter(["Test ", "response"]))
    return agent


@pytest.fixture
def mock_stt():
    """Create a mock STT adapter."""
    with patch('app.services.voice_session.MMSAdapter') as MockSTT:
        stt = MockSTT.return_value
        stt.transcribe = Mock(return_value="test transcript")
        yield stt


@pytest.fixture
def mock_tts():
    """Create a mock TTS provider."""
    with patch('app.services.voice_session.create_tts_provider') as mock_create:
        tts = Mock()
        tts.sample_rate = 24000
        tts.synthesize_streaming = Mock(return_value=iter([b"audio1", b"audio2"]))
        mock_create.return_value = tts
        yield tts


@pytest.fixture
def voice_service(mock_agent, mock_stt, mock_tts):
    """Create a voice session service for testing."""
    service = VoiceSessionService(
        agent=mock_agent,
        stt_provider="mms",
        tts_provider="sesame"
    )
    return service


class TestVoiceSessionService:
    """Test suite for VoiceSessionService."""
    
    def test_initialization(self, voice_service):
        """Test service initialization."""
        assert voice_service.sample_rate == 16000
        assert voice_service.vad_threshold == 0.01
        assert len(voice_service.audio_buffer) == 0
        assert voice_service.silence_frames == 0
    
    def test_calculate_rms(self, voice_service):
        """Test RMS calculation."""
        # Silent audio
        silent = np.zeros(1000)
        rms_silent = voice_service._calculate_rms(silent)
        assert rms_silent == 0.0
        
        # Loud audio
        loud = np.ones(1000) * 0.5
        rms_loud = voice_service._calculate_rms(loud)
        assert rms_loud == 0.5
    
    def test_detect_speech(self, voice_service):
        """Test VAD (voice activity detection)."""
        # Silent audio
        silent = np.zeros(1000)
        assert not voice_service._detect_speech(silent)
        
        # Speech audio (above threshold)
        speech = np.ones(1000) * 0.05  # Above 0.01 threshold
        assert voice_service._detect_speech(speech)
    
    def test_add_audio_chunk(self, voice_service, mock_stt):
        """Test adding audio chunks."""
        # Create PCM audio chunk (16-bit signed int)
        audio_np = np.ones(1600) * 0.1  # 100ms at 16kHz
        audio_int16 = (audio_np * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Add first chunk
        result = voice_service.add_audio_chunk(audio_bytes)
        assert len(voice_service.audio_buffer) == 1
        
        # Add more chunks to trigger STT
        for _ in range(10):
            result = voice_service.add_audio_chunk(audio_bytes)
        
        # Should have called STT
        assert mock_stt.transcribe.called
    
    def test_phrase_boundary_detection(self, voice_service):
        """Test phrase boundary detection."""
        # Add silent chunks
        silent = np.zeros(1600).astype(np.int16)
        silent_bytes = silent.tobytes()
        
        assert not voice_service.check_phrase_boundary()
        
        # Add enough silent chunks to trigger boundary
        for _ in range(voice_service.silence_threshold_frames + 1):
            voice_service.add_audio_chunk(silent_bytes)
        
        assert voice_service.check_phrase_boundary()
    
    def test_get_final_transcript(self, voice_service, mock_stt):
        """Test final transcript extraction."""
        # Add some audio
        audio_np = np.ones(1600) * 0.1
        audio_int16 = (audio_np * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        voice_service.add_audio_chunk(audio_bytes)
        
        # Get final transcript
        transcript = voice_service.get_final_transcript()
        
        assert mock_stt.transcribe.called
        assert transcript == "test transcript"
        assert len(voice_service.audio_buffer) == 0
        assert voice_service.silence_frames == 0
    
    @pytest.mark.asyncio
    async def test_process_with_agent_streaming(self, voice_service, mock_agent, mock_tts):
        """Test processing with streaming agent."""
        transcript = "Hello, how are you?"
        
        responses = []
        async for response in voice_service.process_with_agent(
            transcript=transcript,
            chat_id=1,
            use_streaming=True
        ):
            responses.append(response)
        
        # Check responses
        assert any(r["type"] == "text_start" for r in responses)
        assert any(r["type"] == "text_chunk" for r in responses)
        assert any(r["type"] == "text_complete" for r in responses)
        assert any(r["type"] == "audio_chunk" for r in responses)
        assert any(r["type"] == "audio_complete" for r in responses)
        
        # Check agent was called
        mock_agent.stream_complete_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_with_agent_non_streaming(self, voice_service, mock_agent, mock_tts):
        """Test processing with non-streaming agent."""
        # Make streaming raise NotImplementedError
        mock_agent.stream_complete_text.side_effect = NotImplementedError()
        
        transcript = "Hello, how are you?"
        
        responses = []
        async for response in voice_service.process_with_agent(
            transcript=transcript,
            chat_id=1,
            use_streaming=True
        ):
            responses.append(response)
        
        # Should fall back to complete_text
        mock_agent.complete_text.assert_called_once()
        assert any(r["type"] == "text_complete" for r in responses)
    
    def test_reset(self, voice_service):
        """Test session reset."""
        # Add some state
        audio_np = np.ones(1600) * 0.1
        audio_int16 = (audio_np * 32768).astype(np.int16)
        voice_service.add_audio_chunk(audio_int16.tobytes())
        voice_service.transcript_buffer = "test"
        voice_service.silence_frames = 5
        
        # Reset
        voice_service.reset()
        
        assert len(voice_service.audio_buffer) == 0
        assert voice_service.transcript_buffer == ""
        assert voice_service.silence_frames == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, voice_service, mock_agent):
        """Test error handling in agent processing."""
        mock_agent.stream_complete_text.side_effect = Exception("Test error")
        mock_agent.complete_text.side_effect = Exception("Test error")
        
        responses = []
        async for response in voice_service.process_with_agent(
            transcript="test",
            use_streaming=True
        ):
            responses.append(response)
        
        # Should yield error response
        assert any(r["type"] == "error" for r in responses)
        error_response = next(r for r in responses if r["type"] == "error")
        assert "Test error" in error_response["message"]


class TestVoiceSessionServiceIntegration:
    """Integration tests for voice session workflow."""
    
    def test_full_audio_to_transcript_flow(self, voice_service, mock_stt):
        """Test full flow from audio to transcript."""
        # Simulate receiving audio chunks
        audio_np = np.ones(1600) * 0.1
        audio_int16 = (audio_np * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Add speech chunks
        for _ in range(10):
            voice_service.add_audio_chunk(audio_bytes)
        
        # Add silence to trigger boundary
        silent = np.zeros(1600).astype(np.int16)
        for _ in range(voice_service.silence_threshold_frames + 1):
            voice_service.add_audio_chunk(silent.tobytes())
        
        # Check boundary detected
        assert voice_service.check_phrase_boundary()
        
        # Get final transcript
        transcript = voice_service.get_final_transcript()
        assert transcript == "test transcript"

