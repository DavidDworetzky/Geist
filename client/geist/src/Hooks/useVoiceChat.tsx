import { useState, useRef, useCallback, useEffect } from 'react';
import { LiveVoiceSession } from '../Utils/liveVoiceSession';
import { LocalVoiceSession } from '../Utils/localVoiceSession';
import { DictationSession } from '../Utils/dictationSession';
import { CONVERSATION_PROVIDERS, PipelineVoiceSession } from '../Utils/pipelineVoiceSession';

interface UseVoiceChatProps {
  sessionId: number;
  mode?: 'dictation' | 'live';
  agentType?: string;
  sttProvider?: string;
  ttsProvider?: string;
  ttsModel?: string;
  ttsVoice?: string;
  ttsLanguage?: string;
  onTranscriptFinal?: (text: string) => void;
  onError?: (error: string) => void;
}

type VoiceSession = LiveVoiceSession | LocalVoiceSession | DictationSession | PipelineVoiceSession;

const LISTENING = 'Live · Listening';

const useVoiceChat = (props: UseVoiceChatProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('');
  const [partialTranscript, setPartialTranscript] = useState('');
  const [assistantText, setAssistantText] = useState('');
  const [audioLevel, setAudioLevel] = useState(0);
  const sessionRef = useRef<VoiceSession | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;

  const startRecording = useCallback(async () => {
    if (sessionRef.current) return;
    const current = propsRef.current;
    setIsRecording(true);
    setIsProcessing(true);
    setStatus('Connecting…');
    setPartialTranscript('');
    setAssistantText('');
    let session: VoiceSession;
    const onClosed = () => {
      sessionRef.current = null;
      setIsRecording(false);
      setIsProcessing(false);
      setAudioLevel(0);
      setStatus('');
    };
    const onError = (error: string) => propsRef.current.onError?.(error);
    const onReady = () => { setIsProcessing(false); setStatus(LISTENING); };
    if (current.mode === 'live' && current.ttsProvider && !CONVERSATION_PROVIDERS.has(current.ttsProvider)) {
      session = new PipelineVoiceSession({
        sessionId: current.sessionId,
        agentType: current.agentType,
        sttProvider: current.sttProvider,
        ttsProvider: current.ttsProvider,
        ttsModel: current.ttsModel,
        ttsVoice: current.ttsVoice,
        ttsLanguage: current.ttsLanguage,
        onReady,
        onUserText: setPartialTranscript,
        onAssistantText: setAssistantText,
        onProcessing: processing => {
          setIsProcessing(processing);
          setStatus(processing ? 'Thinking…' : LISTENING);
        },
        onError,
        onClosed
      });
    } else if (current.mode === 'live') {
      const LiveSession = current.ttsProvider === 'local_live' ? LocalVoiceSession : LiveVoiceSession;
      session = new LiveSession({
        model: current.ttsModel,
        voice: current.ttsVoice,
        onReady,
        onTranscript: (role, text) => {
          if (role === 'user') setPartialTranscript(previous => (previous + text).slice(-4000));
          else setAssistantText(previous => (previous + text).slice(-4000));
        },
        onAudioLevel: setAudioLevel,
        onError,
        onClosed
      });
    } else {
      session = new DictationSession({
        provider: current.sttProvider || 'mms',
        onReady: () => { setIsProcessing(false); setStatus('Dictating…'); },
        onProcessing: () => { setIsRecording(false); setIsProcessing(true); setStatus('Transcribing…'); },
        onText: text => propsRef.current.onTranscriptFinal?.(text),
        onError,
        onClosed
      });
    }
    sessionRef.current = session;
    await session.start();
  }, []);

  const stopRecording = useCallback(() => {
    const session = sessionRef.current;
    if (session instanceof DictationSession) session.stop();
    else if (session) { setStatus('Ending call…'); session.close(); }
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
