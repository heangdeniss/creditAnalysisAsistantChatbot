import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { predict, explainShap, fetchDashboardStats, fetchMetricsSummary } from '../api/predict';
import { streamQuery } from '../api/chat';
import RiskDashboard from './RiskDashboard';
import ScenarioPanel from './ScenarioPanel';

const INIT = {
  person_age:                 '',
  person_income:              '',
  person_home_ownership:      'MORTGAGE',
  person_emp_length:          '',
  loan_intent:                'DEBTCONSOLIDATION',
  loan_amnt:                  '',
  loan_int_rate:              '',
  cb_person_default_on_file:  'N',
  cb_person_cred_hist_length: '',
};

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  catboost:            'CatBoost',
  neural_network:      'Neural Network',   // NumPy MLP (15 -> 64 -> 32 -> 1)
  random_forest:       'Random Forest',
};

function asNumber(value, fallback = null) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatCurrency(value, digits = 0) {
  const num = asNumber(value);
  if (num === null) return 'N/A';
  return `$${num.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export default function PredictPanel({ llmModel = 'llama-1b', theme = 'dark' }) {
  const [form,       setForm]       = useState(INIT);
  const [result,     setResult]     = useState(null);
  const [submittedInput, setSubmittedInput] = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState('');
  const [activeChat, setActiveChat] = useState(null); // { modelKey, modelLabel, r }
  const [savedChats, setSavedChats] = useState({});    // { [modelKey]: messages[] }
  const [shapOpen,   setShapOpen]   = useState(null);  // null | modelKey string
  const [shapData,   setShapData]   = useState({});    // { [modelKey]: shap result }
  const [shapError,  setShapError]  = useState({});    // { [modelKey]: string }
  const [dashboardStats,   setDashboardStats]   = useState(null);
  const [metricsSummary,   setMetricsSummary]   = useState(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [dashboardError,   setDashboardError]   = useState('');

  const loadDashboardStats = useCallback(async () => {
    setDashboardLoading(true);
    setDashboardError('');
    try {
      const [stats, metrics] = await Promise.all([
        fetchDashboardStats(),
        fetchMetricsSummary(),
      ]);
      setDashboardStats(stats);
      setMetricsSummary(metrics);
    } catch (err) {
      setDashboardError(err.message ?? 'Failed to load dashboard stats.');
    } finally {
      setDashboardLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDashboardStats();
  }, [loadDashboardStats]);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  async function handleShapToggle(key) {
    const nowOpen = shapOpen === key ? null : key;
    setShapOpen(nowOpen);
    if (nowOpen) {
      // Scroll to SHAP section after React renders it
      setTimeout(() => {
        document.getElementById('shap-section')
          ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 80);
    }
    // Fetch SHAP values on first open only
    if (nowOpen && !shapData[key] && !shapError[key]) {
      try {
        const payload = {
          ...form,
          person_age:                 Number(form.person_age),
          person_income:              Number(form.person_income),
          person_emp_length:          Number(form.person_emp_length),
          loan_amnt:                  Number(form.loan_amnt),
          loan_int_rate:              Number(form.loan_int_rate),
          cb_person_cred_hist_length: Number(form.cb_person_cred_hist_length),
        };
        const data = await explainShap(payload, key);
        setShapData(prev => ({ ...prev, [key]: data }));
      } catch (err) {
        setShapError(prev => ({ ...prev, [key]: err.message }));
      }
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    setSubmittedInput(null);
    try {
      const payload = {
        ...form,
        person_age:                 Number(form.person_age),
        person_income:              Number(form.person_income),
        person_emp_length:          Number(form.person_emp_length),
        loan_amnt:                  Number(form.loan_amnt),
        loan_int_rate:              Number(form.loan_int_rate),
        cb_person_cred_hist_length: Number(form.cb_person_cred_hist_length),
      };
      const newResult = await predict(payload);
      // Clear saved chats only after the new result arrives so the
      // "View saved explanation" hints don't flicker during the request.
      setSavedChats({});
      setShapOpen(null);
      setShapData({});
      setShapError({});
      setResult(newResult);
      setSubmittedInput(payload);
      void loadDashboardStats();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="predict-page">
      <div className="predict-layout">

        {/* ── Form ──────────────────────────────────────────── */}
        <form className="predict-form" onSubmit={handleSubmit}>
          <p className="predict-section-title">Borrower Info</p>

          <div className="predict-grid">
            <Field label="Age (years)">
              <input type="number" min="18" max="100" required
                value={form.person_age} onChange={e => set('person_age', e.target.value)} />
            </Field>

            <Field label="Annual Income ($)">
              <input type="number" min="0" required
                value={form.person_income} onChange={e => set('person_income', e.target.value)} />
            </Field>

            <Field label="Employment Length (years)">
              <input type="number" min="0" max="60" step="0.5" required
                value={form.person_emp_length} onChange={e => set('person_emp_length', e.target.value)} />
            </Field>

            <Field label="Credit History Length (years)">
              <input type="number" min="0" max="60" required
                value={form.cb_person_cred_hist_length} onChange={e => set('cb_person_cred_hist_length', e.target.value)} />
            </Field>

            <Field label="Home Ownership">
              <select value={form.person_home_ownership} onChange={e => set('person_home_ownership', e.target.value)}>
                {['MORTGAGE','OWN','RENT','OTHER'].map(v => <option key={v}>{v}</option>)}
              </select>
            </Field>

            <Field label="Prior Default on File">
              <select value={form.cb_person_default_on_file} onChange={e => set('cb_person_default_on_file', e.target.value)}>
                <option value="N">No</option>
                <option value="Y">Yes</option>
              </select>
            </Field>
          </div>

          <p className="predict-section-title" style={{ marginTop: '20px' }}>Loan Info</p>

          <div className="predict-grid">
            <Field label="Loan Amount ($)">
              <input type="number" min="500" required
                value={form.loan_amnt} onChange={e => set('loan_amnt', e.target.value)} />
            </Field>

            <Field label="Interest Rate (%)">
              <input type="number" min="1" max="40" step="0.01" required
                value={form.loan_int_rate} onChange={e => set('loan_int_rate', e.target.value)} />
            </Field>

            <Field label="Loan Purpose">
              <select value={form.loan_intent} onChange={e => set('loan_intent', e.target.value)}>
                {['DEBTCONSOLIDATION','EDUCATION','HOMEIMPROVEMENT','MEDICAL','PERSONAL','VENTURE']
                  .map(v => <option key={v} value={v}>{v.charAt(0) + v.slice(1).toLowerCase()}</option>)}
              </select>
            </Field>
          </div>

          <button className="predict-btn" type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Credit Risk'}
          </button>

          {error && (
            <p className="predict-error">
              Error: {error}
              <span className="predict-error-hint">Check that the backend is running on port 8000 and try again.</span>
            </p>
          )}
        </form>

        <div className="predict-results">
          <p className="predict-section-title">Model Results</p>

          {!result && !loading && (
            <p className="predict-placeholder">Fill in the form and click Predict to see results.</p>
          )}

          {loading && (
            <div className="predict-placeholder">
              <span className="typing"><span /><span /><span /></span>
              <span style={{ marginLeft: 10 }}>Running models...</span>
            </div>
          )}

          {result && Object.entries(MODEL_LABELS).map(([key, label]) => {
            const r = result[key];
            const clickable = r && !r.error;
            const isShapOpen = shapOpen === key;
            const expectedLoss = r?.expected_loss ?? null;
            const suggestion = r?.approval_suggestion ?? null;
            const lossAmount = asNumber(expectedLoss?.amount);
            const stressedLoss = asNumber(expectedLoss?.stressed_amount);
            const stressedPd = asNumber(expectedLoss?.stressed_pd_pct);
            const lgd = asNumber(expectedLoss?.lgd);
            const suggestedAmount = asNumber(suggestion?.suggested_loan_amount);
            const targetPd = asNumber(suggestion?.target_pd_pct);
            const estimatedPd = asNumber(suggestion?.estimated_pd_pct);
            const capLoan = asNumber(suggestion?.cap_loan_amount);
            return (
              <div key={key}>
                <div
                  className={`result-card ${r ? (r.prediction === 1 ? 'result-default' : 'result-safe') : 'result-unavailable'} ${clickable ? 'result-card-clickable' : ''}`}
                  onClick={() => clickable && setActiveChat({ modelKey: key, modelLabel: label, r })}
                  title={clickable ? 'Click to see explanation' : ''}
                >
                  <div className="result-model-name">{label}</div>
                  {!r ? (
                    <div className="result-na">Model not available</div>
                  ) : r.error ? (
                    <div className="result-na">Error: {r.error}</div>
                  ) : (
                    <>
                      <div className="result-label">{r.label}</div>
                      <div className="result-meta">
                        <span className={`result-grade grade-${r.grade}`}>{r.grade}</span>
                        <span className={`result-decision decision-${r.decision}`}>{r.decision}</span>
                        <span className="result-pct">Calibrated {r.probability.toFixed(1)}% PD</span>
                      </div>
                      <div className="result-submeta">
                        <span>Raw {r.raw_probability?.toFixed(1) ?? r.probability.toFixed(1)}% PD</span>
                        {Array.isArray(r.confidence_band)
                          ? <span>Band {Number(r.confidence_band[0]).toFixed(1)}% - {Number(r.confidence_band[1]).toFixed(1)}%</span>
                          : <span>Band N/A</span>
                        }
                      </div>
                      {expectedLoss && (
                        <div className="result-submeta">
                          {lossAmount !== null && (
                            <span>Expected loss {formatCurrency(lossAmount, 2)} (base)</span>
                          )}
                          {stressedLoss !== null && stressedLoss !== lossAmount && (
                            <span>
                              Stressed {formatCurrency(stressedLoss, 2)}{stressedPd !== null ? ` (PD ${stressedPd.toFixed(1)}%)` : ''}
                            </span>
                          )}
                          {lgd !== null && (
                            <span>LGD {(lgd * 100).toFixed(0)}%</span>
                          )}
                        </div>
                      )}
                      {suggestion && (
                        <div className="result-submeta">
                          {suggestion.status === 'approved_as_is' && suggestedAmount !== null && (
                            <span>
                              Suggested loan {formatCurrency(suggestedAmount)} (meets {(targetPd ?? 0).toFixed(1)}% PD)
                            </span>
                          )}
                          {suggestion.status === 'increase_amount' && suggestedAmount !== null && (
                            <span>
                              Suggested max loan {formatCurrency(suggestedAmount)} to stay under {(targetPd ?? 0).toFixed(1)}% PD
                            </span>
                          )}
                          {suggestion.status === 'cap_reached' && suggestedAmount !== null && (
                            <span>
                              Suggested max loan {formatCurrency(suggestedAmount)} (cap) under {(targetPd ?? 0).toFixed(1)}% PD
                            </span>
                          )}
                          {suggestion.status === 'reduce_amount' && suggestedAmount !== null && (
                            <span>
                              Suggested loan {formatCurrency(suggestedAmount)} to reach {(targetPd ?? 0).toFixed(1)}% PD
                            </span>
                          )}
                          {suggestion.status === 'unreachable' && (
                            <span>Cannot reach {(targetPd ?? 0).toFixed(1)}% PD at minimum loan</span>
                          )}
                          {suggestion.status === 'unavailable' && (
                            <span>Loan suggestion unavailable</span>
                          )}
                          {capLoan !== null && suggestion.status === 'cap_reached' && (
                            <span>Cap {formatCurrency(capLoan)}</span>
                          )}
                          {estimatedPd !== null && !['unavailable', 'unreachable'].includes(suggestion.status) && (
                            <span>Est. PD {estimatedPd.toFixed(1)}%</span>
                          )}
                        </div>
                      )}
                      <div className="result-prob">
                        <div className="result-prob-bar">
                          <div className="result-prob-fill" style={{ width: `${Math.min(r.probability, 100)}%` }} />
                        </div>
                      </div>
                      <ExplainabilityMini result={r} />
                      <div className="result-card-footer">
                        {savedChats[key]?.length > 0
                          ? <div className="result-ask-hint result-ask-saved">View saved explanation</div>
                          : <div className="result-ask-hint">Click to ask why</div>
                        }
                        <button
                            className={`shap-toggle-btn${isShapOpen ? ' shap-toggle-btn-active' : ''}`}
                            onClick={e => { e.stopPropagation(); handleShapToggle(key); }}
                          >
                            {isShapOpen ? 'Hide SHAP' : 'SHAP'}
                          </button>
                      </div>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>

      </div>

      <ScenarioPanel baseApplicant={submittedInput} />

      <RiskDashboard
        result={result}
        inputData={submittedInput}
        dashboardStats={dashboardStats}
        metricsSummary={metricsSummary}
        loading={dashboardLoading}
        error={dashboardError}
        onRefresh={() => { void loadDashboardStats(); }}
        theme={theme}
      />

      {/* ── Full-width SHAP explanation section ──────────── */}
      {result && shapOpen && MODEL_LABELS[shapOpen] && (
        <div className="shap-fw-wrap" id="shap-section">
          <ShapChart
            data={shapData[shapOpen]}
            error={shapError[shapOpen]}
            modelLabel={MODEL_LABELS[shapOpen]}
            standalone
          />
        </div>
      )}

      {/* ── Result Chat Popup ─────────────────────────────── */}
      {activeChat && (
        <ResultChatPopup
          llmModel={llmModel}
          modelLabel={activeChat.modelLabel}
          r={activeChat.r}
          form={form}
          savedMessages={savedChats[activeChat.modelKey] ?? []}
          onSaveMessages={msgs => setSavedChats(prev => ({ ...prev, [activeChat.modelKey]: msgs }))}
          onClose={() => setActiveChat(null)}
        />
      )}

    </div>
  );
}

function ResultChatPopup({ llmModel, modelLabel, r, form, savedMessages, onSaveMessages, onClose }) {
  const [messages,   setMessages]   = useState(savedMessages ?? []);
  const [input,      setInput]      = useState('');
  const [generating, setGenerating] = useState(false);
  const [queued,     setQueued]     = useState(false);
  const abortRef  = useRef(null);
  const bottomRef = useRef(null);
  const didInit   = useRef(false);

  /* ── helpers ── */
  const pctIncome = form.person_income
    ? ((Number(form.loan_amnt) / Number(form.person_income)) * 100).toFixed(1)
    : 'N/A';

  // Brief model description keeps the LLM focused on borrower data, not model internals
  const MODEL_DESCRIPTIONS = {
    logistic_regression: 'a statistical model that scores credit risk using weighted borrower features',
    catboost:            'a gradient-boosted decision tree model trained on credit risk data',
    neural_network:      'a statistical model trained on credit risk data that evaluates borrower characteristics',
    random_forest:       'an ensemble tree model trained on credit risk data',
  };
  const modelKey  = Object.keys(MODEL_LABELS).find(k => MODEL_LABELS[k] === modelLabel) ?? '';
  const modelDesc = MODEL_DESCRIPTIONS[modelKey] ?? 'a credit risk scoring model';

  // Exact borrower data block — sent as pinned facts so LLM cannot substitute example values
  const facts =
    `Model: ${modelLabel} (${modelDesc})
` +
    `Decision: ${r.decision} | Grade: ${r.grade} | Probability of Default: ${r.probability.toFixed(2)}%\n` +
    `Label: ${r.label}\n\n` +
    `Borrower:\n` +
    `  Age: ${form.person_age} years\n` +
    `  Annual Income: $${Number(form.person_income).toLocaleString()}\n` +
    `  Home Ownership: ${form.person_home_ownership}\n` +
    `  Employment Length: ${form.person_emp_length} years\n` +
    `  Prior Default on File: ${form.cb_person_default_on_file === 'Y' ? 'Yes' : 'No'}\n` +
    `  Credit History Length: ${form.cb_person_cred_hist_length} years\n\n` +
    `Loan:\n` +
    `  Amount: $${Number(form.loan_amnt).toLocaleString()}\n` +
    `  Interest Rate: ${form.loan_int_rate}%\n` +
    `  Purpose: ${form.loan_intent}\n` +
    `  Loan-to-Income Ratio: ${pctIncome}%`;

  // Question only — no data embedded (data travels via the facts field)
  const makeQuestion = (userQ) => userQ;

  /* ── stream helper ── */
  function runStream(displayText, questionText) {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setGenerating(true);
    setMessages(prev => [...prev,
      { role: 'user', content: displayText },
      { role: 'bot',  content: '' },
    ]);
    setInput('');
    let acc = '';
    const history = messages
      .filter(m => m.content)
      .slice(-12)
      .map(m => ({ role: m.role === 'bot' ? 'assistant' : 'user', content: m.content }));
    streamQuery(questionText, {
      onQueued: () => setQueued(true),
      onToken: (tok) => {
        setQueued(false);
        acc += tok;
        setMessages(prev => {
          const u = [...prev];
          u[u.length - 1] = { role: 'bot', content: acc };
          return u;
        });
      },
      onDone:  () => { setGenerating(false); setQueued(false); },
      onError: (e) => {
        setGenerating(false);
        setQueued(false);
        setMessages(prev => { const u=[...prev]; u[u.length-1]={role:'bot',content:`Error: ${e}`}; return u; });
      },
      onAbort: () => { setGenerating(false); setQueued(false); },
    }, { signal: ctrl.signal, facts, history, model: llmModel });
  }

  /* ── auto-generate explanation on first open ── */
  useEffect(() => {
    if (didInit.current) return;
    if ((savedMessages ?? []).length > 0) { didInit.current = true; return; }
    didInit.current = true;

    const verb = r.decision === 'APPROVE' ? 'approved'
               : r.decision === 'REJECT'  ? 'rejected'
               : 'flagged for review';

    const display = `Why was this loan ${verb} by ${modelLabel}?`;
    const extra   = r.decision === 'REJECT'
      ? ' After explaining the reasons, suggest specific steps this applicant can take to get approved in the future.'
      : r.decision === 'REVIEW'
      ? ' After explaining, describe what additional conditions or improvements could convert this to a full approval.'
      : ' Highlight the key strengths that led to this positive outcome.';

    const question =
      `Explain clearly and in detail why the ${modelLabel} model ${verb} this loan application. ` +
      `Go through each factor in the EXACT DATA and explain how it contributed to the ${r.decision} decision.${extra}`;

    runStream(display, question);
  // run once on mount — deps intentionally omitted
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── scroll to bottom ── */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  /* ── persist to parent ── */
  useEffect(() => {
    if (messages.length > 0) onSaveMessages(messages);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages]);

  const textareaRef = useRef(null);

  const handleSend = () => {
    const text = input.trim();
    if (!text || generating) return;
    // Reset textarea height before clearing value
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    runStream(text, makeQuestion(text));
  };

  const handleStop = () => abortRef.current?.abort();

  // Abort any in-flight stream before closing
  const handleClose = () => {
    abortRef.current?.abort();
    onClose();
  };

  return (
    <div className="rchat-backdrop" onClick={e => e.target === e.currentTarget && handleClose()}>
      <div className="rchat-panel">
        {/* Header */}
        <div className="rchat-header">
          <div className="rchat-header-info">
            <span className="rchat-header-model">{modelLabel}</span>
            <span className={`rchat-header-decision decision-${r.decision}`}>{r.decision}</span>
            <span className="rchat-header-pd">{r.probability.toFixed(1)}% PD</span>
          </div>
          <button className="rchat-close" onClick={handleClose}>x</button>
        </div>

        {/* Messages */}
        <div className="rchat-messages">
          {messages.map((m, i) => (
            <div key={i} className={`rchat-msg rchat-msg-${m.role}`}>
              {m.role === 'bot' && !m.content && generating
                ? queued
                  ? <span className="queued-msg">Waiting for model...</span>
                  : <span className="typing"><span /><span /><span /></span>
                : m.role === 'bot'
                  ? <div className="rchat-msg-text rchat-md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown></div>
                  : <span className="rchat-msg-text">{m.content}</span>
              }
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="rchat-input-row">
          <textarea
            ref={textareaRef}
            className="rchat-input"
            placeholder="Ask a follow-up question..."
            value={input}
            rows={1}
            onChange={e => {
              setInput(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px';
            }}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
            disabled={generating}
          />
          {generating
            ? <button className="rchat-stop-btn" onClick={handleStop}>Stop</button>
            : <button className="rchat-send-btn" onClick={handleSend} disabled={!input.trim()}>Send</button>
          }
        </div>
      </div>
    </div>
  );
}

// Converts log-odds to probability (needed for waterfall axis positions).
const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));

function ShapChart({ data, error, modelLabel, standalone = false }) {
  if (error) {
    return (
      <div className={standalone ? 'shap-sa-msg shap-sa-error' : 'shap-panel shap-error-panel'}>
        Error: {error}
      </div>
    );
  }
  if (!data) {
    return (
      <div className={standalone ? 'shap-sa-msg shap-sa-loading' : 'shap-panel shap-loading-panel'}>
        <span className="typing"><span /><span /><span /></span>
        <span>Computing SHAP values...</span>
      </div>
    );
  }

  /* ── SVG layout constants (viewBox px) ─────────────────────── */
  const LW     = 180;             // label column width
  const CW     = 630;             // chart track width
  const VW     = 90;              // value column width
  const TW     = LW + CW + VW;   // total SVG width = 900
  const ROW_H  = 40;              // feature row height
  const BAR_H  = 17;              // bar height
  const SPEC_H = 34;              // Base / f(x) row height
  const DOT_R  = 6;               // Base / f(x) dot radius
  const AXIS_H = 28;              // axis area height
  const PAD_T  = 12;
  const PAD_B  = 6;

  /* ── Data ──────────────────────────────────────────────────── */
  const SHOW = 7;
  const top  = data.shap_values.slice(0, SHOW);
  const rest = data.shap_values.slice(SHOW);
  const rows = rest.length > 0
    ? [...top, {
        feature:      '__others__',
        display_name: `+${rest.length} other${rest.length > 1 ? 's' : ''}`,
        shap_value:   rest.reduce((s, e) => s + e.shap_value, 0),
      }]
    : top;

  const usesPercentagePoints = data.model === 'random_forest';
  const clo        = [data.base_value];
  rows.forEach(r => clo.push(clo.at(-1) + r.shap_value));
  const probs      = usesPercentagePoints
    ? clo.map(value => Math.min(1, Math.max(0, value / 100)))
    : clo.map(sigmoid);
  const base_prob  = probs[0];
  const final_prob = probs[probs.length - 1];
  const valueUnit  = usesPercentagePoints ? 'percentage points' : 'log-odds';

  /* ── X mapping: probability → chart pixel ───────────────────── */
  const XPAD   = 0.04;
  const x_min  = Math.max(0, Math.min(...probs) - XPAD);
  const x_max  = Math.min(1, Math.max(...probs) + XPAD);
  const xrange = Math.max(x_max - x_min, 0.01);
  const toX    = p => LW + ((Math.min(1, Math.max(0, p)) - x_min) / xrange) * CW;

  /* ── Y positions ────────────────────────────────────────────── */
  const BASE_Y  = PAD_T;
  const rowTop  = i => PAD_T + SPEC_H + i * ROW_H;
  const FINAL_Y = PAD_T + SPEC_H + rows.length * ROW_H;
  const AXIS_Y  = FINAL_Y + SPEC_H;
  const TOTAL_H = AXIS_Y + AXIS_H + PAD_B;
  const midY    = (t, h) => t + h / 2;

  /* ── Axis ticks: base, 50% (if visible), final ──────────────── */
  const tickSet = new Set([base_prob, final_prob]);
  if (x_min < 0.5 && x_max > 0.5) tickSet.add(0.5);
  const ticks = [...tickSet]
    .sort((a, b) => a - b)
    .filter((t, i, arr) => arr.every((o, j) => j === i || Math.abs(toX(t) - toX(o)) > 30));

  const delta_pct = (final_prob - base_prob) * 100;

  const chart = (
    <>
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="shap-header">
        <div className="shap-title-row">
          <span className="shap-title">SHAP Waterfall</span>
          {modelLabel && <span className="shap-model-chip">{modelLabel}</span>}
        </div>
        <span className="shap-subtitle">
          Base <strong>{(base_prob * 100).toFixed(1)}%</strong>
          {' -> '}f(x) <strong>{(final_prob * 100).toFixed(1)}%</strong>
          {'  '}
          <span className={delta_pct > 0 ? 'shap-legend-pos' : 'shap-legend-neg'}>
            {delta_pct > 0 ? '+' : '-'} {Math.abs(delta_pct).toFixed(1)}% net
          </span>
          {'  |  '}
          <span className="shap-legend-pos">risk+</span>{' '}
          <span className="shap-legend-neg">risk-</span>
        </span>
      </div>

      {/* ── SVG waterfall chart ─────────────────────────────────── */}
      <svg
        className="shap-svg"
        viewBox={`0 0 ${TW} ${TOTAL_H}`}
        width="100%"
        style={{ display: 'block', overflow: 'visible' }}
        aria-label="SHAP waterfall chart"
      >
        <defs>
          {/* Gradient fills for positive / negative bars */}
          <linearGradient id="shap-gp" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#fca5a5" />
            <stop offset="100%" stopColor="#dc2626" />
          </linearGradient>
          <linearGradient id="shap-gn" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#6ee7b7" />
            <stop offset="100%" stopColor="#059669" />
          </linearGradient>
          {/* Diagonal stripe fill for the bundled "others" bar */}
          <pattern id="shap-hatch" width="5" height="5"
            patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" stroke="#94a3b8" strokeWidth="1.8" />
          </pattern>
        </defs>

        {/* 50% risk threshold dashed guideline */}
        {x_min < 0.5 && x_max > 0.5 && (
          <g>
            <line
              x1={toX(0.5)} y1={PAD_T}
              x2={toX(0.5)} y2={AXIS_Y - 2}
              stroke="#3b4880" strokeWidth={1} strokeDasharray="3 3"
            />
            <text x={toX(0.5)} y={PAD_T - 2} textAnchor="middle" className="shap-svg-threshold">
              50%
            </text>
          </g>
        )}

        {/* ── Base row ────────────────────────────────────────── */}
        <text
          x={LW - 8} y={midY(BASE_Y, SPEC_H)}
          textAnchor="end" dominantBaseline="middle"
          className="shap-svg-spec"
        >Base</text>
        <circle
          cx={toX(base_prob)} cy={midY(BASE_Y, SPEC_H)}
          r={DOT_R} className="shap-svg-dot-base"
        />
        <text
          x={LW + CW + 7} y={midY(BASE_Y, SPEC_H)}
          textAnchor="start" dominantBaseline="middle"
          className="shap-svg-val-muted"
        >{(base_prob * 100).toFixed(1)}%</text>

        {/* ── Feature rows ────────────────────────────────────── */}
        {rows.map((r, i) => {
          const sy      = rowTop(i);
          const cym     = midY(sy, ROW_H);
          const prevCym = i === 0 ? midY(BASE_Y, SPEC_H) : midY(rowTop(i - 1), ROW_H);
          const startP  = probs[i];
          const endP    = probs[i + 1];
          const pos     = r.shap_value > 0;
          const bx      = toX(Math.min(startP, endP));
          const bw      = Math.max(Math.abs(toX(endP) - toX(startP)), 2);
          const delta   = (endP - startP) * 100;
          return (
            <g key={r.feature} className="shap-row-g">
              {/* Alternating row tint for readability */}
              {i % 2 === 0 && (
                <rect x={0} y={sy} width={TW} height={ROW_H} className="shap-row-bg" />
              )}
              {/* Dashed connector spanning from previous bar end to this bar start */}
              <line
                x1={toX(startP)} y1={prevCym}
                x2={toX(startP)} y2={cym}
                strokeDasharray="2 2" className="shap-svg-conn"
              />
              {/* Feature label */}
              <text
                x={LW - 8} y={cym}
                textAnchor="end" dominantBaseline="middle"
                className="shap-svg-label"
              >{r.display_name}</text>
              {/* Floating bar */}
              <rect
                x={bx} y={sy + (ROW_H - BAR_H) / 2}
                width={bw} height={BAR_H}
                fill={r.feature === '__others__' ? 'url(#shap-hatch)'
                      : pos ? 'url(#shap-gp)' : 'url(#shap-gn)'}
                rx={3} className="shap-svg-bar"
              >
                <title>
                  {r.display_name}: {delta > 0 ? '+' : ''}{delta.toFixed(2)}%
                  {` (${valueUnit} `}{r.shap_value > 0 ? '+' : ''}{r.shap_value.toFixed(4)}{')'}
                </title>
              </rect>
              {/* Probability delta value */}
              <text
                x={LW + CW + 7} y={cym}
                textAnchor="start" dominantBaseline="middle"
                className={`shap-svg-val ${pos ? 'shap-svg-val-pos' : 'shap-svg-val-neg'}`}
              >{delta > 0 ? '+' : ''}{delta.toFixed(1)}%</text>
            </g>
          );
        })}

        {/* ── f(x) row ────────────────────────────────────────── */}
        {(() => {
          const cym_f  = midY(FINAL_Y, SPEC_H);
          const prevCym = rows.length > 0
            ? midY(rowTop(rows.length - 1), ROW_H)
            : midY(BASE_Y, SPEC_H);
          const valCls = final_prob >= 0.5 ? 'shap-svg-val-pos' : 'shap-svg-val-neg';
          return (
            <g>
              <line
                x1={toX(final_prob)} y1={prevCym}
                x2={toX(final_prob)} y2={cym_f}
                strokeDasharray="2 2" className="shap-svg-conn"
              />
              <text
                x={LW - 8} y={cym_f}
                textAnchor="end" dominantBaseline="middle"
                className="shap-svg-spec"
              >f(x)</text>
              <circle
                cx={toX(final_prob)} cy={cym_f}
                r={DOT_R} className="shap-svg-dot-final"
              />
              <text
                x={LW + CW + 7} y={cym_f}
                textAnchor="start" dominantBaseline="middle"
                className={`shap-svg-val ${valCls}`}
              >{(final_prob * 100).toFixed(1)}%</text>
            </g>
          );
        })()}

        {/* ── X-axis ──────────────────────────────────────────── */}
        <line
          x1={LW} y1={AXIS_Y} x2={LW + CW} y2={AXIS_Y}
          className="shap-svg-axis-line"
        />
        {ticks.map(t => (
          <g key={t}>
            <line
              x1={toX(t)} y1={AXIS_Y}
              x2={toX(t)} y2={AXIS_Y + 4}
              className="shap-svg-axis-tick"
            />
            <text
              x={toX(t)} y={AXIS_Y + 6}
              textAnchor="middle" dominantBaseline="hanging"
              className="shap-svg-axis-text"
            >{(t * 100).toFixed(0)}%</text>
          </g>
        ))}
      </svg>
    </>
  );
  return standalone ? chart : <div className="shap-panel">{chart}</div>;
}

function Field({ label, children }) {
  return (
    <div className="predict-field">
      <label className="predict-label">{label}</label>
      {children}
    </div>
  );
}

function ExplainabilityMini({ result }) {
  const band = Array.isArray(result.confidence_band) ? result.confidence_band : null;
  const features = Array.isArray(result.top_features) ? result.top_features.slice(0, 3) : [];
  if (!band && features.length === 0) return null;

  return (
    <div className="explain-mini">
      {band && (
        <div className="explain-band">
          <span>Confidence band</span>
          <strong>{Number(band[0]).toFixed(1)}% - {Number(band[1]).toFixed(1)}%</strong>
        </div>
      )}
      {features.length > 0 && (
        <div className="explain-feature-list">
          {features.map(feature => (
            <div className="explain-feature-row" key={`${feature.feature}-${feature.direction}`}>
              <span className={`explain-dir explain-dir-${feature.direction}`}>{feature.direction === 'up' ? 'Risk up' : 'Risk down'}</span>
              <span className="explain-feature-name">{feature.display_name ?? feature.feature}</span>
              <strong>{Number(feature.magnitude).toFixed(2)}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
