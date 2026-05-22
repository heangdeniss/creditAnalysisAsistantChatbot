import { useEffect, useMemo, useState } from 'react';
import { simulateScenarios } from '../api/predict';

const EMPTY_DRAFT = {
  name: '',
  person_age: '',
  person_income: '',
  person_home_ownership: '',
  person_emp_length: '',
  loan_intent: '',
  loan_amnt: '',
  loan_int_rate: '',
  cb_person_default_on_file: '',
  cb_person_cred_hist_length: '',
};

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  catboost: 'CatBoost',
  neural_network: 'Neural Network',
};

const FIELD_LABELS = {
  person_age: 'Age',
  person_income: 'Annual Income',
  person_home_ownership: 'Home Ownership',
  person_emp_length: 'Employment Length',
  loan_intent: 'Loan Purpose',
  loan_amnt: 'Loan Amount',
  loan_int_rate: 'Interest Rate',
  cb_person_default_on_file: 'Prior Default',
  cb_person_cred_hist_length: 'Credit History',
};

const NUMERIC_FIELDS = [
  'person_age',
  'person_income',
  'person_emp_length',
  'cb_person_cred_hist_length',
  'loan_amnt',
  'loan_int_rate',
];

function scenarioId() {
  return `scenario-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function round2(value) {
  return Math.round(Number(value) * 100) / 100;
}

function numericValue(value) {
  if (value === '' || value === null || value === undefined) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function buildOverrides(draft) {
  const overrides = {};
  for (const field of NUMERIC_FIELDS) {
    const parsed = numericValue(draft[field]);
    if (parsed !== undefined) overrides[field] = parsed;
  }
  for (const field of ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']) {
    if (draft[field]) overrides[field] = draft[field];
  }
  return overrides;
}

function formatCurrency(value) {
  return `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function formatOverride(field, value) {
  return `${FIELD_LABELS[field] ?? field} ${formatValue(field, value)}`;
}

function formatValue(field, value) {
  if (field === 'person_income' || field === 'loan_amnt') return formatCurrency(value);
  if (field === 'loan_int_rate') return `${Number(value).toFixed(2)}%`;
  if (field === 'person_age' || field === 'person_emp_length' || field === 'cb_person_cred_hist_length') {
    return `${Number(value).toFixed(1)} yrs`;
  }
  if (field === 'cb_person_default_on_file') return value === 'Y' ? 'Yes' : 'No';
  return `${value}`;
}

function formatChange(change) {
  const label = FIELD_LABELS[change.field] ?? change.field;
  return `${label}: ${formatValue(change.field, change.from)} -> ${formatValue(change.field, change.to)}`;
}

function formatProbability(row) {
  if (!row) return 'N/A';
  if (row.error) return 'Error';
  return `${Number(row.probability).toFixed(1)}%`;
}

function formatComparison(baseline, scenario) {
  if (!baseline || !scenario) return 'N/A';
  if (baseline.error || scenario.error) return 'Error';
  return `${Number(baseline.probability).toFixed(1)}% -> ${Number(scenario.probability).toFixed(1)}%`;
}

function deltaClass(delta) {
  if (!delta || delta.status !== 'ok') return 'scenario-delta-flat';
  if (delta.probability_delta > 0) return 'scenario-delta-up';
  if (delta.probability_delta < 0) return 'scenario-delta-down';
  return 'scenario-delta-flat';
}

function formatDelta(delta) {
  if (!delta || delta.status !== 'ok') return 'N/A';
  const value = Number(delta.probability_delta);
  return `${value > 0 ? '+' : ''}${value.toFixed(2)} pts`;
}

function ScoreBadge({ score }) {
  if (!score) return <span className="scenario-score-muted">Unavailable</span>;
  if (score.error) return <span className="scenario-score-error">{score.error}</span>;
  return (
    <span className={`scenario-decision decision-${score.decision}`}>
      {score.decision}
      <span>{score.grade}</span>
    </span>
  );
}

export default function ScenarioPanel({ baseApplicant }) {
  const [draft, setDraft] = useState(EMPTY_DRAFT);
  const [scenarios, setScenarios] = useState([]);
  const [selectedModel, setSelectedModel] = useState('catboost');
  const [includeTopDrivers, setIncludeTopDrivers] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setResult(null);
    setError('');
  }, [baseApplicant]);

  const draftOverrides = useMemo(() => buildOverrides(draft), [draft]);
  const canSave = Boolean(baseApplicant && draft.name.trim() && Object.keys(draftOverrides).length);
  const canRun = Boolean(baseApplicant && scenarios.length && !loading);

  const updateDraft = (field, value) => {
    setDraft(prev => ({ ...prev, [field]: value }));
  };

  const saveScenario = () => {
    if (!canSave) return;
    setScenarios(prev => [
      ...prev,
      {
        id: scenarioId(),
        name: draft.name.trim(),
        overrides: draftOverrides,
      },
    ]);
    setDraft(EMPTY_DRAFT);
    setResult(null);
  };

  const addTemplate = (name, overrides) => {
    if (!baseApplicant) return;
    setScenarios(prev => [...prev, { id: scenarioId(), name, overrides }]);
    setResult(null);
  };

  const addIncomeTemplate = () => {
    addTemplate('Increase income 20%', {
      person_income: round2(Number(baseApplicant.person_income) * 1.2),
    });
  };

  const addLoanBurdenTemplate = () => {
    addTemplate('Reduce loan burden 20%', {
      loan_amnt: Math.max(500, round2(Number(baseApplicant.loan_amnt) * 0.8)),
    });
  };

  const addRateTemplate = () => {
    addTemplate('Lower rate 1 pt', {
      loan_int_rate: Math.max(1, round2(Number(baseApplicant.loan_int_rate) - 1)),
    });
  };

  const addCreditTemplate = () => {
    addTemplate('Add 2 yrs credit history', {
      cb_person_cred_hist_length: Math.min(60, round2(Number(baseApplicant.cb_person_cred_hist_length) + 2)),
    });
  };

  const removeScenario = (id) => {
    setScenarios(prev => prev.filter(scenario => scenario.id !== id));
    setResult(null);
  };

  const runScenarios = async () => {
    if (!canRun) return;
    setLoading(true);
    setError('');
    try {
      const payloadScenarios = scenarios.map(scenario => ({
        scenario_id: scenario.id,
        name: scenario.name,
        ...scenario.overrides,
      }));
      const data = await simulateScenarios(baseApplicant, payloadScenarios, {
        includeTopDrivers,
        driversModel: selectedModel,
      });
      setResult(data);
    } catch (err) {
      setError(err.message ?? 'Scenario simulation failed.');
    } finally {
      setLoading(false);
    }
  };

  const baselineScore = result?.baseline?.scores?.[selectedModel];
  const baselineLti = result?.baseline?.derived_metrics?.loan_to_income_pct;

  return (
    <section className="scenario-panel">
      <div className="scenario-head">
        <div>
          <p className="predict-section-title">Scenario Simulator</p>
          <p className="scenario-subtitle">Baseline what-if scoring with local models only.</p>
        </div>
        <div className="scenario-head-controls">
          <label className="scenario-inline-control">
            <span>Model</span>
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
              {Object.entries(MODEL_LABELS).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
          </label>
          <label className="scenario-check">
            <input
              type="checkbox"
              checked={includeTopDrivers}
              onChange={e => setIncludeTopDrivers(e.target.checked)}
            />
            <span>Top drivers</span>
          </label>
        </div>
      </div>

      {!baseApplicant ? (
        <p className="dashboard-empty">Run a prediction first to set the baseline applicant.</p>
      ) : (
        <>
          <div className="scenario-template-row">
            <button type="button" onClick={addIncomeTemplate}>Income +20%</button>
            <button type="button" onClick={addLoanBurdenTemplate}>Loan -20%</button>
            <button type="button" onClick={addRateTemplate}>Rate -1 pt</button>
            <button type="button" onClick={addCreditTemplate}>Credit +2 yrs</button>
          </div>

          <div className="scenario-builder-grid">
            <ScenarioField label="Scenario Name">
              <input
                value={draft.name}
                onChange={e => updateDraft('name', e.target.value)}
                placeholder="Increase income 20%"
                maxLength={120}
              />
            </ScenarioField>
            <ScenarioField label="Annual Income ($)">
              <input type="number" min="0" value={draft.person_income} onChange={e => updateDraft('person_income', e.target.value)} />
            </ScenarioField>
            <ScenarioField label="Loan Amount ($)">
              <input type="number" min="500" value={draft.loan_amnt} onChange={e => updateDraft('loan_amnt', e.target.value)} />
            </ScenarioField>
            <ScenarioField label="Interest Rate (%)">
              <input type="number" min="1" max="40" step="0.01" value={draft.loan_int_rate} onChange={e => updateDraft('loan_int_rate', e.target.value)} />
            </ScenarioField>
            <ScenarioField label="Employment (years)">
              <input type="number" min="0" max="60" step="0.5" value={draft.person_emp_length} onChange={e => updateDraft('person_emp_length', e.target.value)} />
            </ScenarioField>
            <ScenarioField label="Credit History (years)">
              <input type="number" min="0" max="60" value={draft.cb_person_cred_hist_length} onChange={e => updateDraft('cb_person_cred_hist_length', e.target.value)} />
            </ScenarioField>
            <ScenarioField label="Home Ownership">
              <select value={draft.person_home_ownership} onChange={e => updateDraft('person_home_ownership', e.target.value)}>
                <option value="">No change</option>
                {['MORTGAGE', 'OWN', 'RENT', 'OTHER'].map(value => <option key={value} value={value}>{value}</option>)}
              </select>
            </ScenarioField>
            <ScenarioField label="Loan Purpose">
              <select value={draft.loan_intent} onChange={e => updateDraft('loan_intent', e.target.value)}>
                <option value="">No change</option>
                {['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'].map(value => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </ScenarioField>
            <ScenarioField label="Prior Default">
              <select value={draft.cb_person_default_on_file} onChange={e => updateDraft('cb_person_default_on_file', e.target.value)}>
                <option value="">No change</option>
                <option value="N">No</option>
                <option value="Y">Yes</option>
              </select>
            </ScenarioField>
          </div>

          <div className="scenario-actions">
            <button type="button" className="scenario-save-btn" onClick={saveScenario} disabled={!canSave}>Save Scenario</button>
            <button type="button" className="scenario-run-btn" onClick={runScenarios} disabled={!canRun}>
              {loading ? 'Running...' : 'Run Scenarios'}
            </button>
          </div>

          {error ? (
            <p className="predict-error">
              Error: {error}
              <span className="predict-error-hint">Verify the backend is running and the baseline applicant is valid.</span>
            </p>
          ) : null}

          <div className="scenario-saved-list">
            {scenarios.length === 0 ? (
              <p className="dashboard-empty">No saved scenarios yet.</p>
            ) : scenarios.map(scenario => (
              <div className="scenario-saved-row" key={scenario.id}>
                <div>
                  <strong>{scenario.name}</strong>
                  <span>{Object.entries(scenario.overrides).map(([field, value]) => formatOverride(field, value)).join(' | ')}</span>
                </div>
                <button type="button" onClick={() => removeScenario(scenario.id)} aria-label={`Remove ${scenario.name}`}>x</button>
              </div>
            ))}
          </div>

          {result ? (
            <div className="scenario-results">
              <div className="scenario-baseline-row">
                <span>Baseline</span>
                <strong>{formatProbability(baselineScore)}</strong>
                <ScoreBadge score={baselineScore} />
                <em>{baselineLti === null || baselineLti === undefined ? 'LTI N/A' : `LTI ${baselineLti.toFixed(1)}%`}</em>
              </div>

              <div className="scenario-table">
                <div className="scenario-table-head">
                  <span>Scenario</span>
                  <span>Base to Scenario</span>
                  <span>Delta</span>
                  <span>Decision</span>
                </div>
                {result.scenarios.map(scenario => {
                  const score = scenario.scores?.[selectedModel];
                  const delta = scenario.deltas?.[selectedModel];
                  const lti = scenario.derived_metrics?.loan_to_income_pct;
                  const drivers = scenario.top_drivers?.drivers ?? [];
                  const changes = Array.isArray(scenario.changes) ? scenario.changes : [];
                  return (
                    <div className="scenario-result-row" key={scenario.scenario_id ?? scenario.name}>
                      <div>
                        <strong>{scenario.name}</strong>
                        <span>{lti === null || lti === undefined ? 'LTI N/A' : `LTI ${lti.toFixed(1)}%`}</span>
                        {changes.length > 0 ? (
                          <div className="scenario-change-row">
                            {changes.map(change => (
                              <span className="scenario-change-chip" key={`${scenario.scenario_id}-${change.field}`}>
                                {formatChange(change)}
                              </span>
                            ))}
                          </div>
                        ) : null}
                        {drivers.length > 0 ? (
                          <div className="scenario-driver-row">
                            {drivers.slice(0, 3).map(driver => (
                              <span className={`scenario-driver-${driver.direction}`} key={`${scenario.scenario_id}-${driver.feature}`}>
                                {driver.display_name}
                              </span>
                            ))}
                          </div>
                        ) : null}
                        {delta?.impact_summary ? (
                          <div className="scenario-impact">{delta.impact_summary}</div>
                        ) : null}
                      </div>
                      <strong>{formatComparison(baselineScore, score)}</strong>
                      <span className={deltaClass(delta)}>{formatDelta(delta)}</span>
                      <ScoreBadge score={score} />
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

function ScenarioField({ label, children }) {
  return (
    <div className="scenario-field">
      <label>{label}</label>
      {children}
    </div>
  );
}
