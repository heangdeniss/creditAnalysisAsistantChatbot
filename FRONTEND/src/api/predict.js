const API = '/api';

const _timeout = (ms, ctrl) => setTimeout(() => ctrl.abort(), ms);

/**
 * Send borrower data to the backend and receive credit-risk predictions
 * from all available ML models.
 *
 * @param {object} data  Raw borrower features
 * @returns {Promise<object>}  { logistic_regression, catboost, neural_network, random_forest }
 */
export async function predict(data) {
  const controller = new AbortController();
  const timer = _timeout(30_000, controller);

  try {
    const res = await fetch(`${API}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
      body:    JSON.stringify(data),
    });

    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(detail ?? res.statusText);
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Prediction timed out (30 s).');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Score a baseline borrower plus saved what-if scenarios.
 *
 * @param {object} baseApplicant  Raw borrower features
 * @param {Array<object>} scenarios  [{ scenario_id, name, ...overrides }]
 * @param {object} options  { includeTopDrivers, driversModel }
 * @returns {Promise<object>}  { baseline, scenarios, field_notes }
 */
export async function simulateScenarios(baseApplicant, scenarios, options = {}) {
  const controller = new AbortController();
  const timer = _timeout(30_000, controller);

  const body = {
    base_applicant: baseApplicant,
    scenarios,
    include_top_drivers: Boolean(options.includeTopDrivers),
    drivers_model: options.driversModel ?? 'catboost',
  };

  try {
    const res = await fetch(`${API}/scenario`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
      body:    JSON.stringify(body),
    });

    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(Array.isArray(detail) ? detail.map(d => d.msg).join('; ') : detail ?? res.statusText);
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Scenario simulation timed out (30 s).');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch SHAP feature attributions for one borrower from the chosen model.
 *
 * @param {object} data   Borrower features (same shape as predict())
 * @param {string} model  'catboost' | 'logistic_regression' | 'neural_network' | 'random_forest'
 * @returns {Promise<object>}  { model, base_value, shap_values: [...] }
 */
export async function explainShap(data, model = 'catboost') {
  const controller = new AbortController();
  const timer = _timeout(30_000, controller);

  try {
    const res = await fetch(`${API}/explain?model=${encodeURIComponent(model)}`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
      body:    JSON.stringify(data),
    });

    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(detail ?? res.statusText);
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('SHAP request timed out (30 s).');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch aggregate corpus and retrieval statistics for the dashboard.
 *
 * @param {number} limit Maximum number of documents to sample
 * @returns {Promise<object>} { documents: {...}, retrieval: {...} }
 */
export async function fetchDashboardStats(limit = 1500) {
  const controller = new AbortController();
  const timer = _timeout(30_000, controller);

  try {
    const res = await fetch(`${API}/dashboard/stats?limit=${encodeURIComponent(limit)}`, {
      method:  'GET',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
    });

    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(detail ?? res.statusText);
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Dashboard stats request timed out (30 s).');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch recent backend observability metrics.
 *
 * @returns {Promise<object>} summary from /metrics/summary
 */
export async function fetchMetricsSummary() {
  const controller = new AbortController();
  const timer = _timeout(30_000, controller);

  try {
    const res = await fetch(`${API}/metrics/summary`, {
      method:  'GET',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
    });

    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(detail ?? res.statusText);
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Metrics request timed out (30 s).');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch contextual-bandit learner status.
 *
 * @returns {Promise<object>} policy learner summary
 */
export async function fetchPolicySummary() {
  const controller = new AbortController();
  const timer = _timeout(30_000, controller);

  try {
    const res = await fetch(`${API}/policy/summary`, {
      method:  'GET',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
    });

    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(detail ?? res.statusText);
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Policy summary timed out (30 s).');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Send realized outcome feedback to train the contextual-bandit policy.
 *
 * @param {object} feedback Includes action, applicant, and either reward or outcome fields
 * @returns {Promise<object>} learner update result
 */
export async function submitPolicyFeedback(feedback) {
  const controller = new AbortController();
  const timer = _timeout(30_000, controller);

  try {
    const res = await fetch(`${API}/policy/feedback`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
      body:    JSON.stringify(feedback),
    });

    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(Array.isArray(detail) ? detail.map(d => d.msg).join('; ') : detail ?? res.statusText);
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Policy feedback timed out (30 s).');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
