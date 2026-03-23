import { useRef, useState } from 'react';
import { ocrAnalyzeStream } from '../api/ocr';

const ACCEPTED = ['image/png','image/jpeg','image/jpg','image/webp','image/bmp','image/tiff'];

// stage machine: idle | ocr | llama | done | error
export default function OCRPanel({ onAskAbout }) {
  const [preview,   setPreview]   = useState(null);
  const [file,      setFile]      = useState(null);
  const [ocrResult, setOcrResult] = useState(null);
  const [analysis,  setAnalysis]  = useState('');
  const [stage,     setStage]     = useState('idle');
  const [error,     setError]     = useState('');
  const [copied,    setCopied]    = useState(false);
  const [dragging,  setDragging]  = useState(false);
  const inputRef = useRef();

  function acceptFile(f) {
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) { setError('Unsupported file type. Please upload PNG, JPEG, WEBP, BMP or TIFF.'); return; }
    if (f.size > 20 * 1024 * 1024)  { setError('File too large (max 20 MB).'); return; }
    setError(''); setOcrResult(null); setAnalysis(''); setStage('idle'); setCopied(false);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
    setFile(f);
  }

  const onDrop      = e => { e.preventDefault(); setDragging(false); acceptFile(e.dataTransfer.files[0]); };
  const onDragOver  = e => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  function handleClear() {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null); setFile(null); setOcrResult(null);
    setAnalysis(''); setStage('idle'); setError(''); setCopied(false);
  }

  async function handleAnalyze() {
    if (!file) return;
    setStage('ocr'); setError(''); setOcrResult(null); setAnalysis(''); setCopied(false);
    await ocrAnalyzeStream(file, {
      onOcrText: payload => { setOcrResult(payload); setStage('llama'); },
      onToken:   token   => setAnalysis(prev => prev + token),
      onDone:    ()      => setStage('done'),
      onError:   err     => { setError(err.message); setStage('error'); },
    });
  }

  async function handleCopy(text) {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const busy = stage === 'ocr' || stage === 'llama';

  return (
    <div className="ocr-page">
      <div className="ocr-layout">

        {/* ── Left: upload ──────────────────────────────────────── */}
        <div className="ocr-left">
          <p className="predict-section-title">📄 Upload Document Image</p>
          <p className="ocr-hint">
            Upload a scan or photo of any credit document — pay stub, bank statement,
            loan application — to extract text with EasyOCR, then analyse it with
            Llama&nbsp;3.2&nbsp;1B.
          </p>

          <div
            className={`ocr-dropzone ${dragging ? 'ocr-dropzone-active' : ''} ${preview ? 'ocr-dropzone-has-image' : ''}`}
            onClick={() => !preview && inputRef.current?.click()}
            onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
          >
            {preview
              ? <img src={preview} alt="preview" className="ocr-preview-img" />
              : <div className="ocr-dropzone-empty">
                  <span className="ocr-upload-icon">🖼️</span>
                  <p className="ocr-upload-label">Drag & drop or click to upload</p>
                  <p className="ocr-upload-sub">PNG · JPEG · WEBP · BMP · TIFF · max 20 MB</p>
                </div>
            }
          </div>

          <input ref={inputRef} type="file" accept={ACCEPTED.join(',')}
            style={{ display: 'none' }} onChange={e => acceptFile(e.target.files[0])} />

          <div className="ocr-actions">
            {preview && <>
              <button className="ocr-btn ocr-btn-secondary" onClick={() => inputRef.current?.click()} disabled={busy}>🔄 Change</button>
              <button className="ocr-btn ocr-btn-secondary" onClick={handleClear} disabled={busy}>🗑️ Clear</button>
            </>}
            <button className="ocr-btn ocr-btn-primary" onClick={handleAnalyze} disabled={!file || busy}>
              {stage === 'ocr'   ? '⏳ Running OCR…'
               : stage === 'llama' ? '🦙 Llama analysing…'
               : '🔍 Extract & Analyse'}
            </button>
          </div>

          {error && <p className="predict-error">⚠️ {error}</p>}

          {/* pipeline stage indicators */}
          <div className="ocr-pipeline">
            <Step label="EasyOCR"       status={stageOf('ocr',   stage)} />
            <span className="ocr-pipe-arrow">→</span>
            <Step label="Llama 3.2 1B" status={stageOf('llama', stage)} />
          </div>

          {ocrResult && (
            <div className="ocr-stats">
              <span>🔤 {ocrResult.word_count} words</span>
              <span>⚡ {ocrResult.duration_s}s OCR</span>
              <span>📦 {ocrResult.blocks.length} blocks</span>
            </div>
          )}
        </div>

        {/* ── Right: results ────────────────────────────────────── */}
        <div className="ocr-right">
          <p className="predict-section-title">🦙 Llama 3.2 1B Analysis</p>

          {stage === 'idle' && !analysis && (
            <div className="ocr-placeholder">
              <span className="ocr-placeholder-icon">🦙</span>
              <p>Llama will analyse the extracted text for credit risk factors.</p>
            </div>
          )}

          {stage === 'ocr' && (
            <div className="ocr-placeholder">
              <span className="typing"><span /><span /><span /></span>
              <p style={{ marginTop: 12 }}>Stage 1 — EasyOCR extracting text…</p>
            </div>
          )}

          {(stage === 'llama' || stage === 'done' || analysis) && (
            <div className="ocr-analysis-box">
              {analysis
                ? analysis.split('\n').map((line, i) => <p key={i} className="ocr-text-line">{line}</p>)
                : <><span className="typing"><span /><span /><span /></span>
                    <p style={{ marginTop: 10, color: 'var(--muted)', fontSize: '0.84rem' }}>Stage 2 — Llama generating analysis…</p></>
              }
            </div>
          )}

          {analysis && (
            <div className="ocr-result-actions" style={{ marginBottom: 20 }}>
              <button className="ocr-btn ocr-btn-secondary" onClick={() => handleCopy(analysis)}>
                {copied ? '✅ Copied!' : '📋 Copy Analysis'}
              </button>
              {onAskAbout && ocrResult?.text && (
                <button className="ocr-btn ocr-btn-primary"
                  onClick={() => onAskAbout(`Analyse this document for credit risk.\n\nExtracted text:\n${ocrResult.text}`)}>
                  💬 Ask More in Chat
                </button>
              )}
            </div>
          )}

          {/* raw OCR text — collapsible */}
          {ocrResult && (
            <details className="ocr-blocks-details">
              <summary className="ocr-blocks-summary">
                📝 Raw OCR text ({ocrResult.word_count} words)
              </summary>
              <div className="ocr-text-box" style={{ marginTop: 10 }}>
                {ocrResult.text
                  ? ocrResult.text.split('\n').map((l, i) => <p key={i} className="ocr-text-line">{l}</p>)
                  : <p className="ocr-no-text">No text detected.</p>
                }
              </div>
              {ocrResult.blocks.length > 0 && (
                <div className="ocr-blocks-list" style={{ marginTop: 8 }}>
                  {ocrResult.blocks.map((b, i) => (
                    <div key={i} className="ocr-block-row">
                      <span className="ocr-block-text">{b.text}</span>
                      <span className="ocr-block-conf"
                        style={{ color: b.confidence > 0.8 ? 'var(--green)' : b.confidence > 0.5 ? 'var(--yellow)' : 'var(--error)' }}>
                        {(b.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </details>
          )}
        </div>
      </div>
    </div>
  );
}

function stageOf(step, current) {
  const order = ['idle', 'ocr', 'llama', 'done'];
  const si = order.indexOf(step);
  const ci = order.indexOf(current === 'error' ? 'idle' : current);
  if (current === 'error') return 'error';
  if (si < ci)  return 'done';
  if (si === ci) return 'active';
  return 'pending';
}

function Step({ label, status }) {
  const icon = { active: '⏳', done: '✅', pending: '⬜', error: '❌' }[status] ?? '⬜';
  return (
    <div className={`ocr-step ocr-step-${status}`}>
      <span>{icon}</span>
      <span>{label}</span>
    </div>
  );
}
