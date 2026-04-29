/**
 * chat.js — API helpers for the Credit Risk RAG backend (FastAPI :8000)
 *
 * streamQuery()  → SSE streaming (primary path)
 * sendQuery()    → blocking fallback
 * checkHealth()  → readiness ping
 */

const API = '/api'; // proxied by Vite to http://localhost:8000

const timeout = (ms, controller) => setTimeout(() => controller.abort(), ms);

/**
 * Stream tokens from POST /api/stream via Server-Sent Events.
 *
 * @param {string} question
 * @param {{ onToken?: fn, onContext?: fn, onDone?: fn, onError?: fn }} cbs
 * @param {{ signal?: AbortSignal, facts?: string, history?: Array, model?: string }} opts
 */
export async function streamQuery(question, cbs = {}, { signal: externalSignal, facts = '', history = [], model = 'llama-1b' } = {}) {
  const controller = new AbortController();
  externalSignal?.addEventListener('abort', () => controller.abort());
  const timer = timeout(600_000, controller);

  try {
    const res = await fetch(`${API}/stream`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
      body: JSON.stringify({ question, facts, history, model }),
    });

    if (!res.ok) {
      const { detail, error } = await res.json().catch(() => ({}));
      throw new Error(detail ?? error ?? res.statusText);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { done, value } = await reader.read();
      // Flush any remaining buffered data on stream close
      if (done) {
        if (buf.trimEnd()) {
          const tail = buf.startsWith('data: ') ? buf : '';
          if (tail) try {
            const msg = JSON.parse(tail.slice(6));
            if (msg.type === 'context') cbs.onContext?.(msg.chunks);
            if (msg.type === 'token')   cbs.onToken?.(msg.text);
            if (msg.type === 'queued')  cbs.onQueued?.();
            if (msg.type === 'error')   cbs.onError?.(new Error(msg.message || 'Stream failed.'));
            if (msg.type === 'done')    cbs.onDone?.();
          } catch { /* skip malformed */ }
        }
        break;
      }

      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop(); // keep incomplete tail

      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        try {
          const msg = JSON.parse(part.slice(6));
          if (msg.type === 'context') cbs.onContext?.(msg.chunks);
          if (msg.type === 'token')   cbs.onToken?.(msg.text);
          if (msg.type === 'queued')  cbs.onQueued?.();
          if (msg.type === 'error')   cbs.onError?.(new Error(msg.message || 'Stream failed.'));
          if (msg.type === 'done')    cbs.onDone?.();
        } catch { /* skip malformed */ }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') cbs.onAbort?.();
    else cbs.onError?.(err);
  } finally {
    clearTimeout(timer);
  }
}

/** Blocking POST /api/query — returns { question, answer, context }. */
export async function sendQuery(question, { history = [], model = 'llama-1b' } = {}) {
  const controller = new AbortController();
  const timer = timeout(600_000, controller);

  try {
    const res = await fetch(`${API}/query`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
      body: JSON.stringify({ question, history, model }),
    });

    if (!res.ok) {
      const { detail, error } = await res.json().catch(() => ({}));
      throw new Error(detail ?? error ?? res.statusText);
    }

    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ping the backend health endpoint.
 * Returns { ok: bool, busy: bool } — busy is true when the model is mid-generation.
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    if (!res.ok) return { ok: false, busy: false };
    const { busy } = await res.json().catch(() => ({}));
    return { ok: true, busy: !!busy };
  } catch {
    return { ok: false, busy: false };
  }
}

/** Switch active chat model and force backend to preload it immediately. */
export async function switchModel(model) {
  const controller = new AbortController();
  const timer = timeout(600_000, controller);

  try {
    const res = await fetch(`${API}/model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({ model }),
    });

    if (!res.ok) {
      const { detail, error } = await res.json().catch(() => ({}));
      throw new Error(detail ?? error ?? res.statusText);
    }

    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/** Upload a WAV audio blob and get back transcribed text from whisper-medium. */
export async function transcribeAudio(audioBlob, { language = 'en' } = {}) {
  const controller = new AbortController();
  const timer = timeout(600_000, controller);

  try {
    const form = new FormData();
    form.append('file', audioBlob, 'microphone.wav');
    form.append('language', language);

    const res = await fetch(`${API}/transcribe`, {
      method: 'POST',
      signal: controller.signal,
      body: form,
    });

    if (!res.ok) {
      const { detail, error } = await res.json().catch(() => ({}));
      throw new Error(detail ?? error ?? res.statusText);
    }

    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}
