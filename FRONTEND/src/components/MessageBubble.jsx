import '../style/MessageBubble.css';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function normalizeChunk(chunk, index) {
  if (typeof chunk === 'string') {
    return {
      citation_id: `S${index + 1}`,
      source: `Chunk ${index + 1}`,
      page: null,
      score: null,
      snippet: chunk,
      text: chunk,
      title: '',
      chunk_id: '',
    };
  }
  return {
    citation_id: chunk.citation_id ?? `S${index + 1}`,
    document_id: chunk.document_id ?? chunk.source ?? 'knowledge-base',
    source: chunk.source ?? 'knowledge-base',
    page: chunk.page ?? null,
    score: chunk.score ?? null,
    snippet: chunk.snippet ?? chunk.text ?? '',
    text: chunk.text ?? chunk.snippet ?? '',
    title: chunk.title ?? '',
    chunk_id: chunk.chunk_id ?? '',
  };
}

export default function MessageBubble({ msg, showContext }) {
  const isUser = msg.role === 'user';
  const chunks = (msg.context ?? []).map(normalizeChunk);

  return (
    <div className={`msg msg-${isUser ? 'user' : 'bot'}`}>
      <div className="avatar">{isUser ? 'You' : 'AI'}</div>

      <div className="bubble-wrap">
        <div className="bubble">
          {msg.loading ? (
            msg.queued ? <span className="queued-msg">Waiting for model...</span> : <Dots />
          ) : isUser ? (
            <p>{msg.content}</p>
          ) : (
            <>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
              {msg.streaming === true && <span className="cursor" />}
            </>
          )}
        </div>

        {msg.stopped && !isUser && (
          <span className="stopped-msg">Stopped</span>
        )}

        {showContext && chunks.length > 0 && (
          <details className="ctx">
            <summary>{chunks.length} cited source{chunks.length > 1 ? 's' : ''} retrieved</summary>
            <div className="ctx-list">
              {chunks.map((c, i) => (
                <div key={i} className="ctx-chunk">
                  <div className="ctx-meta-row">
                    <span className="ctx-num">[{c.citation_id}]</span>
                    <span className="ctx-source">
                      Doc {c.document_id}{c.page !== null ? `, page ${c.page}` : ''}
                    </span>
                    {typeof c.score === 'number' && <span className="ctx-score">{c.score.toFixed(3)}</span>}
                  </div>
                  {c.chunk_id && <small className="ctx-id">Chunk: {c.chunk_id}</small>}
                  <p>{c.snippet.length > 400 ? `${c.snippet.slice(0, 400)}...` : c.snippet}</p>
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
