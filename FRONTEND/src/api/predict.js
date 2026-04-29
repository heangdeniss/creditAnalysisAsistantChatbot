const API = '/api';

const _timeout = (ms, ctrl) => setTimeout(() => ctrl.abort(), ms);

/**
 * Send borrower data to the backend and receive credit-risk predictions
 * from all available ML models.
 *
 * @param {object} data  Raw borrower features
 * @returns {Promise<object>}  { logistic_regression, catboost }
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
 * Fetch SHAP feature attributions for one borrower from the chosen model.
 *
 * @param {object} data   Borrower features (same shape as predict())
 * @param {string} model  'catboost' | 'logistic_regression'
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
