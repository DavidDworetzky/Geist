import { useState, useRef, useCallback, useEffect } from 'react';
import { LiveVoiceSession } from '../Utils/liveVoiceSession';
import { LocalVoiceSession } from '../Utils/localVoiceSession';
import { DictationSession } from '../Utils/dictationSession';

interface UseVoiceChatProps {
  sessionId: number;
  mode: 'dictation' | 'live';
  sttProvider?: string;
  ttsProvider?: string;
  ttsModel?: string;
  ttsVoice?: string;
  onTranscriptFinal?: (text: string) => void;
  onError?: (error: string) => void;
  onLiveStart?: () => void;
  onLiveDelegate?: (transcript: string, signal: AbortSignal) => Promise<string>;
}

const useVoiceChat = (props: UseVoiceChatProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('');
  const [partialTranscript, setPartialTranscript] = useState('');
  const [assistantText, setAssistantText] = useState('');
  const [audioLevel, setAudioLevel] = useState(0);
  const sessionRef = useRef<LiveVoiceSession | LocalVoiceSession | DictationSession | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;

  const startRecording = useCallback(async () => {
    if (sessionRef.current) return;
    const current = propsRef.current;
    if (current.mode === 'live') current.onLiveStart?.();
    setIsRecording(true);
    setIsProcessing(true);
    setStatus('Connecting…');
    setPartialTranscript('');
    setAssistantText('');
    const onClosed = () => {
      sessionRef.current = null;
      setIsRecording(false);
      setIsProcessing(false);
      setAudioLevel(0);
      setStatus('');
    };
    const onError = (error: string) => propsRef.current.onError?.(error);
    const LiveSession = current.ttsProvider === 'local_live' ? LocalVoiceSession : LiveVoiceSession;
    const session = current.mode === 'live' ? new LiveSession({
      model: current.ttsModel,
      voice: current.ttsVoice,
      onDelegate: current.onLiveDelegate
        ? (transcript, signal) => propsRef.current.onLiveDelegate!(transcript, signal) : undefined,
      onReady: () => { setIsProcessing(false); setStatus('Live · Listening'); },
      onTranscript: (role, text) => {
        if (role === 'user') setPartialTranscript(previous => (previous + text).slice(-4000));
        else setAssistantText(previous => (previous + text).slice(-4000));
      },
      onAudioLevel: setAudioLevel,
      onError,
      onClosed
    }) : new DictationSession({
      provider: current.sttProvider || 'mms',
      onReady: () => { setIsProcessing(false); setStatus('Dictating…'); },
      onProcessing: () => { setIsRecording(false); setIsProcessing(true); setStatus('Transcribing…'); },
      onText: text => propsRef.current.onTranscriptFinal?.(text),
      onError,
      onClosed
    });
    sessionRef.current = session;
    await session.start();
  }, []);

  const stopRecording = useCallback(() => {
    const session = sessionRef.current;
    if (session instanceof LiveVoiceSession || session instanceof LocalVoiceSession) { setStatus('Ending call…'); session.close(); }
    else session?.stop();
  }, []);

  const toggleRecording = useCallback(() => {
    if (sessionRef.current) stopRecording();
    else void startRecording();
  }, [startRecording, stopRecording]);

  useEffect(() => () => sessionRef.current?.dispose(), []);
  useEffect(() => {
    sessionRef.current?.dispose();
    setPartialTranscript('');
    setAssistantText('');
  }, [props.sessionId]);

  return { isRecording, isProcessing, status, partialTranscript, assistantText, audioLevel,
    startRecording, stopRecording, toggleRecording };
};

export default useVoiceChat;
