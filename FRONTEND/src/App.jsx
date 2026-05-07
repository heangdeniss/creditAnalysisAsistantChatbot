import { useCallback, useEffect, useRef, useState } from 'react';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import Sidebar from './components/Sidebar';
import PredictPanel from './components/PredictPanel';
import { streamQuery } from './api/chat';

const MODEL_OPTIONS = [
  { value: 'llama-1b', label: 'Llama 3.2 1B' },
  { value: 'llama-3b', label: 'Llama 3.2 3B' },
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [status,   setStatus]   = useState('idle');
  const [prefill,  setPrefill]  = useState('');
  const [tab,      setTab]      = useState('chat');   // 'chat' | 'predict'
  const [model,    setModel]    = useState('llama-1b');
  const [showContext, setShowContext] = useState(false);
  const msgId       = useRef(0);
  const abortCtrl   = useRef(null);  // holds the AbortController for the current stream
  const messagesRef = useRef([]);    // stable ref so handleSend always sees latest messages

  // Keep messagesRef in sync with state so handleSend (empty deps) can read history
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  // Update browser tab title when switching tabs
  useEffect(() => {
    document.title = tab === 'chat'
      ? 'Chat — Credit Risk Assistant'
      : 'Predict — Credit Risk Assistant';
  }, [tab]);

  const handleStop = useCallback(() => {
    abortCtrl.current?.abort();
  }, []);

  const handleSend = useCallback(async (text) => {
    const uid = ++msgId.current;
    const bid = ++msgId.current;

    setMessages(prev => [
      ...prev,
      { id: uid, role: 'user', content: text },
      { id: bid, role: 'bot',  content: '', context: [], loading: true },
    ]);
    setStatus('busy');

    const ctrl = new AbortController();
    abortCtrl.current = ctrl;

    const patch = (id, delta) =>
      setMessages(prev => prev.map(m =>
        m.id !== id ? m : { ...m, ...(typeof delta === 'function' ? delta(m) : delta) }
      ));

    // Build history from all completed turns (max last 6 exchanges = 12 messages)
    const history = messagesRef.current
      .filter(m => m.content && !m.loading)
      .map(m => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content }))
      .slice(-12);

    await streamQuery(
      text,
      {
        onContext: chunks => patch(bid, { context: chunks }),
        onToken:   token  => patch(bid, m => ({ content: m.content + token, loading: false, streaming: true })),
        onQueued:  ()     => patch(bid, { queued: true }),
        onDone:    ()     => { setStatus('idle');  patch(bid, { streaming: false, queued: false }); abortCtrl.current = null; },
        onAbort:   ()     => { setStatus('idle');  patch(bid, { streaming: false, stopped: true, queued: false }); abortCtrl.current = null; },
        onError:   err    => {
          patch(bid, { content: `⚠️ ${err.message}\n\nTry again or check that the backend is running.`, loading: false });
          setStatus('error');
          abortCtrl.current = null;
        },
      },
      { signal: ctrl.signal, history, model },
    );
  }, [model]);

  const clearPrefill  = useCallback(() => setPrefill(''), []);
  const clearMessages = useCallback(() => { abortCtrl.current?.abort(); setMessages([]); setStatus('idle'); }, []);

  return (
    <div className="layout">
      <Sidebar
        onPromptClick={q => { setPrefill(q); setTab('chat'); }}
        onClear={clearMessages}
        hasMsgs={messages.length > 0}
        activeModel={model}
      />

      <div className="main">
        <header className="topbar">
          <div className="topbar-left">
            <span className="topbar-logo">🦙</span>
            <span className="topbar-title">Credit Risk Assistant</span>
          </div>

          <nav className="tab-bar">
            <button className={`tab-btn ${tab === 'chat' ? 'tab-active' : ''}`}
              onClick={() => setTab('chat')}>💬 Chat</button>
            <button className={`tab-btn ${tab === 'predict' ? 'tab-active' : ''}`}
              onClick={() => setTab('predict')}>🔍 Predict</button>
          </nav>

          <label className="model-switch" title="Choose chat model">
            <span>Model</span>
            <select
              className="model-select"
              value={model}
              onChange={e => setModel(e.target.value)}
              disabled={status === 'busy'}
            >
              {MODEL_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </label>

          <label className="topbar-toggle" title="Show retrieved context">
            <input
              type="checkbox"
              checked={showContext}
              onChange={e => setShowContext(e.target.checked)}
              disabled={status === 'busy'}
            />
            <span>Sources</span>
          </label>

          <span className={`pill pill-${status}`}>
            {{ idle: 'Ready', busy: 'Generating…', error: 'Error' }[status]}
          </span>
        </header>

        {/* Both panels are always mounted — CSS hides the inactive one so state is preserved */}
        <div style={{ display: tab === 'chat' ? 'contents' : 'none' }}>
          <ChatWindow messages={messages} showContext={showContext} />
          <InputBar
            onSend={handleSend}
            onStop={handleStop}
            generating={status === 'busy'}
            disabled={false}
            prefill={prefill}
            onPrefillUsed={clearPrefill}
            onClearError={() => { if (status === 'error') setStatus('idle'); }}
          />
        </div>
        <div style={{ display: tab === 'predict' ? 'contents' : 'none' }}>
          <PredictPanel llmModel={model} />
        </div>
      </div>
    </div>
  );
}
