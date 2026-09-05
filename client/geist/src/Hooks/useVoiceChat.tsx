import { useState, useRef, useCallback, useEffect } from 'react';

interface UseVoiceChatProps {
  sessionId: number;
  agentType?: string;
  sttProvider?: string;
  ttsProvider?: string;
  ttsModel?: string;
  ttsVoice?: string;
  ttsLanguage?: string;
  ttsInstruct?: string;
  ttsSpeed?: number;
  onTranscriptPartial?: (text: string) => void;
  onTranscriptFinal?: (text: string) => void;
  onAssistantText?: (text: string) => void;
  onError?: (error: string) => void;
}

const useVoiceChat = (props: UseVoiceChatProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [partialTranscript, setPartialTranscript] = useState('');
  const [assistantText, setAssistantText] = useState('');
  const callbacks = useRef(props);
  callbacks.current = props;
  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const sourcesRef = useRef(new Set<AudioBufferSourceNode>());
  const nextPlaybackTimeRef = useRef(0);
  const generationRef = useRef(0);
  const activeRef = useRef(false);
  const readyRef = useRef(false);
  const processingRef = useRef(false);

  const clearPlayback = useCallback(() => {
    sourcesRef.current.forEach(source => {
      source.onended = null;
      source.stop();
      source.disconnect();
    });
    sourcesRef.current.clear();
    nextPlaybackTimeRef.current = 0;
  }, []);

  const stopRecording = useCallback(() => {
    generationRef.current += 1;
    activeRef.current = readyRef.current = processingRef.current = false;
    const socket = wsRef.current;
    wsRef.current = null;
    if (socket) {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'stop' }));
      socket.close();
    }
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;
    clearPlayback();
    for (const ref of [audioContextRef, playbackContextRef]) {
      if (ref.current) void ref.current.close();
      ref.current = null;
    }
    setIsRecording(false);
    setIsProcessing(false);
    setPartialTranscript('');
  }, [clearPlayback]);

  const startRecording = useCallback(async () => {
    if (activeRef.current) return;
    activeRef.current = true;
    const generation = ++generationRef.current;
    const current = () => generation === generationRef.current;
    try {
      // Resume playback during the user gesture, not on a network callback.
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const playback = new AudioContextClass();
      playbackContextRef.current = playback;
      await playback.resume();
      if (!current()) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      if (!current()) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      mediaStreamRef.current = stream;
      const capture = new AudioContextClass({ sampleRate: 16000 });
      audioContextRef.current = capture;
      if (capture.sampleRate !== 16000) throw new Error('Microphone must support 16 kHz audio');
      const source = capture.createMediaStreamSource(stream);
      const processor = capture.createScriptProcessor(1024, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (event: AudioProcessingEvent) => {
        const ws = wsRef.current;
        if (!readyRef.current || processingRef.current || sourcesRef.current.size
          || !ws || ws.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          const sample = Math.max(-1, Math.min(1, input[i]));
          pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
        }
        ws.send(pcm.buffer);
      };
      source.connect(processor);
      processor.connect(capture.destination);
      const options = callbacks.current;
      const params = new URLSearchParams({
        session_id: String(options.sessionId), agent_type: options.agentType || 'online',
        stt_provider: options.sttProvider || 'mms', tts_provider: options.ttsProvider || 'sesame',
      });
      for (const [key, value] of Object.entries({
        tts_model: options.ttsModel, tts_voice: options.ttsVoice,
        tts_language: options.ttsLanguage, tts_instruct: options.ttsInstruct,
        tts_speed: options.ttsSpeed,
      })) {
        if (value !== undefined) params.set(key, String(value));
      }
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(`${protocol}//${window.location.host}/api/v1/voice/stream?${params}`);
      socket.binaryType = 'arraybuffer';
      wsRef.current = socket;
      let sampleRate = 0;
      let messages = Promise.resolve();
      socket.onmessage = event => {
        // Preserve frame order across Blob conversion; stop/reconnect invalidates
        // pending callbacks before they can create any more audio.
        messages = messages.then(async () => {
          if (!current()) return;
          const data = event.data instanceof Blob ? await event.data.arrayBuffer() : event.data;
          if (!current()) return;
          if (data instanceof ArrayBuffer) {
            if (!sampleRate || data.byteLength % 2) throw new Error('Invalid voice PCM frame');
            if (!data.byteLength) return;
            const pcm = new Int16Array(data);
            const buffer = playback.createBuffer(1, pcm.length, sampleRate);
            const samples = buffer.getChannelData(0);
            for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
            const source = playback.createBufferSource();
            source.buffer = buffer;
            source.connect(playback.destination);
            sourcesRef.current.add(source);
            source.onended = () => { sourcesRef.current.delete(source); source.disconnect(); };
            const startAt = Math.max(playback.currentTime + 0.02, nextPlaybackTimeRef.current);
            source.start(startAt);
            nextPlaybackTimeRef.current = startAt + buffer.duration;
            return;
          }
          const message = JSON.parse(data);
          switch (message.type) {
            case 'ready': readyRef.current = true; break;
            case 'transcript_partial':
              setPartialTranscript(message.text || '');
              callbacks.current.onTranscriptPartial?.(message.text || ''); break;
            case 'transcript_final':
              setPartialTranscript('');
              callbacks.current.onTranscriptFinal?.(message.text || ''); break;
            case 'processing':
            case 'text_start':
              processingRef.current = true;
              setIsProcessing(true);
              setAssistantText(''); break;
            case 'text_chunk':
              setAssistantText(prev => prev + (message.text || ''));
              callbacks.current.onAssistantText?.(message.text || ''); break;
            case 'text_complete': setAssistantText(message.text || ''); break;
            case 'audio_start':
              if (message.encoding !== 'pcm_s16le' || message.channels !== 1
                || !Number.isInteger(message.sample_rate) || message.sample_rate < 8000
                || message.sample_rate > 96000) throw new Error('Unsupported voice audio format');
              sampleRate = message.sample_rate;
              // Do not rewind: audio from the preceding turn may still be audible.
              break;
            case 'reset_complete':
              clearPlayback();
              sampleRate = 0;
              processingRef.current = false;
              setIsProcessing(false);
              setPartialTranscript(''); break;
            case 'done':
              processingRef.current = false;
              setIsProcessing(false); break;
            case 'error': throw new Error(message.message || 'Voice processing failed');
          }
        }).catch(error => {
          if (!current()) return;
          stopRecording();
          callbacks.current.onError?.(error.message || 'Voice playback failed');
        });
        return messages;
      };
      socket.onerror = () => {
        if (!current()) return;
        stopRecording();
        callbacks.current.onError?.('Voice connection failed');
      };
      socket.onclose = () => { if (current()) stopRecording(); };
      setIsRecording(true);
    } catch (error) {
      if (!current()) return;
      stopRecording();
      callbacks.current.onError?.(error instanceof Error ? error.message : 'Microphone unavailable');
    }
  }, [clearPlayback, stopRecording]);

  const toggleRecording = useCallback(() => {
    if (activeRef.current) stopRecording();
    else void startRecording();
  }, [startRecording, stopRecording]);

  useEffect(() => () => stopRecording(), [stopRecording, props.sessionId]);
  return { isRecording, isProcessing, partialTranscript, assistantText,
    startRecording, stopRecording, toggleRecording };
};

export default useVoiceChat;
