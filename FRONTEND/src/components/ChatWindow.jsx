import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';

export default function ChatWindow({ messages, showContext }) {
  const endRef       = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    // Only auto-scroll if user is already near the bottom (within 120 px)
    if (scrollHeight - scrollTop - clientHeight < 120) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <section className="chat-window" ref={containerRef}>
      {/* Welcome — shown only when the conversation is empty */}
      {messages.length === 0 && (
        <div className="msg msg-bot">
          <div className="avatar">🦙</div>
          <div className="bubble-wrap">
            <div className="bubble">
              👋 Hello! I'm your <strong>credit risk assistant</strong>.<br />
              Ask me anything about credit risk, loan defaults, or financial analysis.
            </div>
          </div>
        </div>
      )}

      {messages.map(m => (
        <MessageBubble key={m.id} msg={m} showContext={showContext} />
      ))}

      <div ref={endRef} />
    </section>
  );
}
