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
      {messages.map(m => (
        <MessageBubble key={m.id} msg={m} showContext={showContext} />
      ))}

      <div ref={endRef} />
    </section>
  );
}
