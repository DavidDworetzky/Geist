import { useState, useRef, useCallback, useEffect } from 'react';

interface VoiceMessage {
  type: string;
  text?: string;
  message?: string;
}

interface UseVoiceChatProps {
  sessionId: number;
  agentType?: string;
  sttProvider?: string;
  ttsProvider?: string;
  onTranscriptPartial?: (text: string) => void;
  onTranscriptFinal?: (text: string) => void;
  onAssistantText?: (text: string) => void;
  onError?: (error: string) => void;
}

const useVoiceChat = ({
  sessionId,
  agentType = 'online',
  sttProvider = 'mms',
  ttsProvider = 'sesame',
  onTranscriptPartial,
  onTranscriptFinal,
  onAssistantText,
  onError
}: UseVoiceChatProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [partialTranscript, setPartialTranscript] = useState('');
  const [assistantText, setAssistantText] = useState('');
  
  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioQueueRef = useRef<ArrayBuffer[]>([]);
  const isPlayingRef = useRef(false);

  // WebSocket setup
  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/voice/stream?session_id=${sessionId}&agent_type=${agentType}&stt_provider=${sttProvider}&tts_provider=${ttsProvider}`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('Voice WebSocket connected');
    };
    
    ws.onmessage = async (event) => {
      if (event.data instanceof Blob) {
        // Audio chunk received
        const arrayBuffer = await event.data.arrayBuffer();
        audioQueueRef.current.push(arrayBuffer);
        if (!isPlayingRef.current) {
          playAudioQueue();
        }
      } else {
        // JSON message
        const message: VoiceMessage = JSON.parse(event.data);
        
        switch (message.type) {
          case 'ready':
            console.log('Voice session ready');
            break;
          case 'transcript_partial':
            setPartialTranscript(message.text || '');
            onTranscriptPartial?.(message.text || '');
            break;
          case 'transcript_final':
            setPartialTranscript('');
            onTranscriptFinal?.(message.text || '');
            break;
          case 'text_start':
            setIsProcessing(true);
            setAssistantText('');
            break;
          case 'text_chunk':
            setAssistantText(prev => prev + (message.text || ''));
            onAssistantText?.(message.text || '');
            break;
          case 'text_complete':
            setAssistantText(message.text || '');
            break;
          case 'audio_chunk_start':
            // Next message will be audio binary
            break;
          case 'audio_complete':
            break;
          case 'done':
            setIsProcessing(false);
            break;
          case 'error':
            console.error('Voice error:', message.message);
            onError?.(message.message || 'Unknown error');
            break;
        }
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      onError?.('WebSocket connection error');
    };
    
    ws.onclose = () => {
      console.log('Voice WebSocket disconnected');
    };
    
    wsRef.current = ws;
    return ws;
  }, [sessionId, agentType, sttProvider, ttsProvider, onTranscriptPartial, onTranscriptFinal, onAssistantText, onError]);

  // Audio playback
  const playAudioQueue = useCallback(async () => {
    if (audioQueueRef.current.length === 0 || isPlayingRef.current) {
      return;
    }
    
    isPlayingRef.current = true;
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    while (audioQueueRef.current.length > 0) {
      const audioBuffer = audioQueueRef.current.shift()!;
      
      // Convert Int16 PCM to Float32 for Web Audio
      const int16Array = new Int16Array(audioBuffer);
      const float32Array = new Float32Array(int16Array.length);
      
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
      }
      
      // Create audio buffer
      const buffer = audioContext.createBuffer(1, float32Array.length, 24000); // 24kHz from Sesame
      buffer.getChannelData(0).set(float32Array);
      
      // Play
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      
      await new Promise<void>(resolve => {
        source.onended = () => resolve();
        source.start();
      });
    }
    
    isPlayingRef.current = false;
  }, []);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }
        
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Convert Float32 to Int16 PCM
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Send to WebSocket
        wsRef.current.send(int16Data.buffer);
      };
      
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      // Connect WebSocket
      connectWebSocket();
      
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      onError?.('Microphone access denied or unavailable');
    }
  }, [connectWebSocket, onError]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsRecording(false);
    setPartialTranscript('');
  }, []);

  // Toggle recording
  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return {
    isRecording,
    isProcessing,
    partialTranscript,
    assistantText,
    startRecording,
    stopRecording,
    toggleRecording
  };
};

export default useVoiceChat;

