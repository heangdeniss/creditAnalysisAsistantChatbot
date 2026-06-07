import '../style/InputBar.css';
import { useEffect, useRef, useState } from 'react';
import { transcribeAudio } from '../api/chat';

function resize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function mergeFloat32Chunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return buffer;
}

export default function InputBar({ onSend, onStop, generating, disabled, prefill, onPrefillUsed, onClearError }) {
  const ref = useRef(null);
  const cbRef = useRef(onPrefillUsed);

  const audioStreamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const sinkRef = useRef(null);
  const audioChunksRef = useRef([]);
  const sampleRateRef = useRef(16000);
  const recordingRef = useRef(false);
  const speechRecRef = useRef(null);
  const liveBaseTextRef = useRef('');
  const liveDraftRef = useRef('');

  const [recording, setRecording] = useState(false);
  const [speechBusy, setSpeechBusy] = useState(false);
  const [speechError, setSpeechError] = useState('');

  // keep cbRef in sync
  useEffect(() => { cbRef.current = onPrefillUsed; });

  useEffect(() => {
    if (!prefill) return;
    const el = ref.current;
    if (!el) return;
    el.value = prefill;
    resize(el);
    el.focus();
    cbRef.current?.();
  }, [prefill]);

  useEffect(() => {
    return () => {
      cleanupCapture();
    };
  }, []);

  function cleanupCapture() {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    sinkRef.current?.disconnect();

    if (audioStreamRef.current) {
      for (const track of audioStreamRef.current.getTracks()) track.stop();
    }

    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
    }

    audioStreamRef.current = null;
    audioCtxRef.current = null;
    sourceRef.current = null;
    processorRef.current = null;
    sinkRef.current = null;
  }

  function stopLiveRecognition() {
    if (speechRecRef.current) {
      try {
        speechRecRef.current.onresult = null;
        speechRecRef.current.onend = null;
        speechRecRef.current.onerror = null;
        speechRecRef.current.stop();
      } catch (err) {
        void err;
        // Ignore browser-specific stop errors
      }
    }
    speechRecRef.current = null;
  }

  function applyLiveTextPreview(spoken) {
    const el = ref.current;
    if (!el) return;

    const base = liveBaseTextRef.current || '';
    const spokenText = (spoken || '').trim();
    const spacer = base && spokenText ? ' ' : '';
    el.value = `${base}${spacer}${spokenText}`;
    resize(el);
  }

  function startLiveRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    try {
      const rec = new SpeechRecognition();
      rec.lang = 'en-US';
      rec.interimResults = true;
      rec.continuous = true;

      rec.onresult = event => {
        let finalText = '';
        let interimText = '';

        for (let i = 0; i < event.results.length; i += 1) {
          const entry = event.results[i];
          const transcript = entry?.[0]?.transcript || '';
          if (entry.isFinal) finalText += `${transcript} `;
          else interimText += transcript;
        }

        liveDraftRef.current = `${finalText}${interimText}`.trim();
        applyLiveTextPreview(liveDraftRef.current);
      };

      rec.onend = () => {
        if (recordingRef.current) {
          try {
            rec.start();
          } catch (err) {
            void err;
          }
        }
      };

      rec.onerror = () => {
        // Keep Whisper final transcription as source of truth.
      };

      rec.start();
      speechRecRef.current = rec;
    } catch (err) {
      void err;
      speechRecRef.current = null;
    }
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setSpeechError('Microphone is not supported in this browser.');
      return;
    }

    try {
      setSpeechError('');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      if (ctx.state === 'suspended') await ctx.resume();

      const sourceNode = ctx.createMediaStreamSource(stream);
      const processorNode = ctx.createScriptProcessor(4096, 1, 1);
      const silentGain = ctx.createGain();
      silentGain.gain.value = 0;

      audioChunksRef.current = [];
      sampleRateRef.current = ctx.sampleRate;
      recordingRef.current = true;
      liveBaseTextRef.current = ref.current?.value || '';
      liveDraftRef.current = '';

      startLiveRecognition();

      processorNode.onaudioprocess = event => {
        const channel = event.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(new Float32Array(channel));
      };

      sourceNode.connect(processorNode);
      processorNode.connect(silentGain);
      silentGain.connect(ctx.destination);

      audioStreamRef.current = stream;
      audioCtxRef.current = ctx;
      sourceRef.current = sourceNode;
      processorRef.current = processorNode;
      sinkRef.current = silentGain;

      setRecording(true);
      onClearError?.();
    } catch (err) {
      recordingRef.current = false;
      stopLiveRecognition();
      cleanupCapture();
      setSpeechError(err?.message || 'Could not access microphone.');
    }
  }

  async function stopRecordingAndTranscribe() {
    setRecording(false);
    recordingRef.current = false;
    setSpeechBusy(true);

    const chunks = audioChunksRef.current.slice();
    const sampleRate = sampleRateRef.current;
    // Always clear live preview before inserting Whisper result
    stopLiveRecognition();
    cleanupCapture();
    liveDraftRef.current = '';

    try {
      const samples = mergeFloat32Chunks(chunks);
      if (!samples.length) {
        throw new Error('No audio captured. Try speaking a little louder.');
      }

      const wavBuffer = encodeWav(samples, sampleRate);
      const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
      const result = await transcribeAudio(wavBlob, { language: 'en' });
      const text = (result?.text || '').trim();

      if (!text) {
        setSpeechError('No speech detected.');
        applyLiveTextPreview('');
        return;
      }

      const el = ref.current;
      if (!el) return;

      // Only use the base text (before live preview) plus Whisper result
      const baseText = liveBaseTextRef.current || '';
      const spacer = baseText && text ? ' ' : '';
      el.value = `${baseText}${spacer}${text}`.trimStart();
      resize(el);
      el.focus();

      setSpeechError('');
      onClearError?.();
    } catch (err) {
      setSpeechError(err?.message || 'Speech-to-text failed.');
      applyLiveTextPreview('');
    } finally {
      setSpeechBusy(false);
      liveDraftRef.current = '';
    }
  }

  async function handleMicClick() {
    if (recording) {
      await stopRecordingAndTranscribe();
      return;
    }
    await startRecording();
  }

  function submit() {
    const text = ref.current?.value.trim();
    if (!text || disabled) return;
    onSend(text);
    ref.current.value = '';
    ref.current.style.height = 'auto';
  }

  const busy = disabled || generating || speechBusy || recording;
  const micDisabled = disabled || generating || speechBusy;
  const statusHint = speechError || (recording ? 'Recording audio...' : speechBusy ? 'Transcribing audio...' : '');

  return (
    <footer className="input-bar">
      <div className="input-row">
        <textarea
          ref={ref}
          className="input-box"
          placeholder="Ask about credit risk..."
          rows={1}
          disabled={disabled}
          onInput={e => { resize(e.target); onClearError?.(); }}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!busy) submit(); } }}
        />
        <button
          className={`mic-btn ${recording ? 'mic-btn-recording' : ''}`}
          onClick={handleMicClick}
          disabled={micDisabled}
          aria-label={recording ? 'Stop recording' : 'Start recording'}
          title={recording ? 'Stop recording' : 'Speak to text'}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="2.5" width="6" height="12" rx="3" />
            <path d="M5 11.5a7 7 0 0 0 14 0" />
            <line x1="12" y1="18.5" x2="12" y2="21.5" />
            <line x1="8.5" y1="21.5" x2="15.5" y2="21.5" />
          </svg>
        </button>
        {generating
          ? (
            <button className="stop-btn" onClick={onStop} aria-label="Stop generating" title="Stop generating">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2"/>
              </svg>
            </button>
          ) : (
            <button className="send-btn" onClick={submit} disabled={busy} aria-label="Send">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          )
        }
      </div>
      {statusHint ? (
        <p className={`input-hint ${speechError ? 'input-hint-error' : ''}`}>{statusHint}</p>
      ) : null}
    </footer>
  );
}
