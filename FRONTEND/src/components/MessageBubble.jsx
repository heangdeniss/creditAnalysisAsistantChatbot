import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function MessageBubble({ msg, showContext }) {
  const isUser = msg.role === 'user';

  return (
    <div className={`msg msg-${isUser ? 'user' : 'bot'}`}>
      <div className="avatar">{isUser ? '👤' : '🦙'}</div>

      <div className="bubble-wrap">
        <div className="bubble">
          {msg.loading ? (
            msg.queued ? <span className="queued-msg">⏳ Waiting for model…</span> : <Dots />
          ) : isUser ? (
            <p>{msg.content}</p>
          ) : (
            <>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
              {/* Blinking cursor while streaming */}
          {msg.streaming === true && <span className="cursor" />}
            </>
          )}
        </div>

        {msg.stopped && !isUser && (
          <span className="stopped-msg">⛔ Stopped</span>
        )}

        {showContext && msg.context?.length > 0 && (
          <details className="ctx">
            <summary>📄 {msg.context.length} chunk{msg.context.length > 1 ? 's' : ''} retrieved</summary>
            <div className="ctx-list">
              {msg.context.map((c, i) => (
                <div key={i} className="ctx-chunk">
                  <span className="ctx-num">#{i + 1}</span>
                  <p>{c.length > 400 ? c.slice(0, 400) + '…' : c}</p>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

function Dots() {
  return (
    <span className="typing">
      <span /><span /><span />
    </span>
  );
}
