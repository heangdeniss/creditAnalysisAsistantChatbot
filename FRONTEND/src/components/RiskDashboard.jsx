import { useMemo } from 'react';
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  RadialLinearScale,
  Tooltip,
} from 'chart.js';
import { Bar, Doughnut, Radar } from 'react-chartjs-2';

ChartJS.register(
  ArcElement,
  BarElement,
  CategoryScale,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  RadialLinearScale,
  Tooltip,
);

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  catboost: 'CatBoost',
  neural_network: 'Neural Network',
};

const DECISION_COLORS = {
  APPROVE: '#3ddb82',
  REVIEW: '#f5c542',
  REJECT: '#ff5f5f',
};

const baseChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 520, easing: 'easeOutQuart' },
  plugins: {
    legend: { labels: { color: '#dde3f5', boxWidth: 12, usePointStyle: true } },
    tooltip: {
      backgroundColor: 'rgba(19,22,32,0.95)',
      borderColor: '#252a40',
      borderWidth: 1,
      titleColor: '#dde3f5',
      bodyColor: '#c7d0ea',
      displayColors: true,
    },
  },
};

function asNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatCount(value) {
  return asNumber(value, 0).toLocaleString();
}

function formatCurrency(value) {
  return `$${asNumber(value, 0).toLocaleString()}`;
}

function formatIntent(value) {
  const raw = String(value ?? '').trim();
  if (!raw) return '-';
  return raw
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\b\w/g, ch => ch.toUpperCase());
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function InputMetric({ label, value }) {
  return (
    <div className="dashboard-input-item">
      <span className="dashboard-input-label">{label}</span>
      <strong className="dashboard-input-value">{value}</strong>
    </div>
  );
}

export default function RiskDashboard({
  result,
  inputData,
  dashboardStats,
  metricsSummary,
  loading = false,
  error = '',
  onRefresh,
}) {
  const modelRows = useMemo(() => {
    if (!result) return [];

    return Object.entries(MODEL_LABELS)
      .map(([key, label]) => {
        const row = result[key];
        if (!row || row.error) return null;
        return {
          key,
          label,
          probability: asNumber(row.probability, 0),
          decision: row.decision ?? 'REVIEW',
          grade: row.grade ?? '-',
        };
      })
      .filter(Boolean);
  }, [result]);

  const borrowerInput = useMemo(() => {
    if (!inputData) return null;

    const income = asNumber(inputData.person_income, 0);
    const loanAmount = asNumber(inputData.loan_amnt, 0);
    const loanToIncome = income > 0 ? (loanAmount / income) * 100 : null;

    return {
      age: asNumber(inputData.person_age, 0),
      income,
      homeOwnership: inputData.person_home_ownership || '-',
      employmentLength: asNumber(inputData.person_emp_length, 0),
      loanIntent: inputData.loan_intent || '-',
      loanAmount,
      interestRate: asNumber(inputData.loan_int_rate, 0),
      priorDefault: inputData.cb_person_default_on_file === 'Y' ? 'Yes' : 'No',
      creditHistoryLength: asNumber(inputData.cb_person_cred_hist_length, 0),
      loanToIncome,
    };
  }, [inputData]);

  const docs = dashboardStats?.documents ?? {};
  const retrieval = dashboardStats?.retrieval ?? {};
  const metrics = metricsSummary ?? {};

  const decisionCounts = useMemo(() => {
    const counts = { APPROVE: 0, REVIEW: 0, REJECT: 0 };
    for (const row of modelRows) {
      if (counts[row.decision] !== undefined) counts[row.decision] += 1;
    }
    return counts;
  }, [modelRows]);

  const averagePd = modelRows.length
    ? modelRows.reduce((sum, row) => sum + row.probability, 0) / modelRows.length
    : 0;

  const highestRisk = modelRows.length
    ? modelRows.reduce((maxRow, row) => (row.probability > maxRow.probability ? row : maxRow), modelRows[0])
    : null;

  const consensus = Object.entries(decisionCounts)
    .sort((a, b) => b[1] - a[1])[0];

  const modelProbabilityData = {
    labels: modelRows.length ? modelRows.map(row => row.label) : ['Awaiting prediction'],
    datasets: [
      {
        label: 'Probability of Default (%)',
        data: modelRows.length ? modelRows.map(row => row.probability) : [0],
        borderRadius: 8,
        backgroundColor: modelRows.length
          ? modelRows.map(row => DECISION_COLORS[row.decision] ?? '#5b7fff')
          : ['#3b4880'],
      },
    ],
  };

  const modelProbabilityOptions = {
    ...baseChartOptions,
    plugins: {
      ...baseChartOptions.plugins,
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { color: '#a9b5d7' },
        grid: { display: false },
      },
      y: {
        min: 0,
        max: 100,
        ticks: {
          color: '#a9b5d7',
          callback: value => `${value}%`,
        },
        grid: { color: 'rgba(96, 106, 138, 0.22)' },
      },
    },
  };

  const inputRiskLabels = ['Age', 'Employment', 'Credit History', 'Interest Rate', 'Loan-to-Income'];
  const hasBorrowerInput = Boolean(borrowerInput);

  const inputRiskScores = hasBorrowerInput
    ? [
        clamp(borrowerInput.age, 0, 100),
        clamp((borrowerInput.employmentLength / 60) * 100, 0, 100),
        clamp((borrowerInput.creditHistoryLength / 60) * 100, 0, 100),
        clamp((borrowerInput.interestRate / 40) * 100, 0, 100),
        clamp(((borrowerInput.loanToIncome ?? 0) / 200) * 100, 0, 100),
      ]
    : [0, 0, 0, 0, 0];

  const inputRiskData = {
    labels: inputRiskLabels,
    datasets: [
      {
        label: 'Borrower Input Pressure',
        data: inputRiskScores,
        borderRadius: 8,
        backgroundColor: 'rgba(91, 127, 255, 0.78)',
        borderColor: '#7b9aff',
        borderWidth: 1,
      },
      {
        type: 'line',
        label: 'Average Model PD',
        data: inputRiskLabels.map(() => Math.min(100, averagePd)),
        borderColor: '#f5c542',
        backgroundColor: 'rgba(245, 197, 66, 0.25)',
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        tension: 0.3,
      },
    ],
  };

  const inputRiskOptions = {
    ...baseChartOptions,
    scales: {
      x: {
        ticks: { color: '#a9b5d7' },
        grid: { display: false },
      },
      y: {
        min: 0,
        max: 100,
        ticks: {
          color: '#a9b5d7',
          callback: value => `${value}%`,
        },
        grid: { color: 'rgba(96, 106, 138, 0.22)' },
      },
    },
  };

  const hasDecisionData = Object.values(decisionCounts).some(v => v > 0);
  const decisionData = {
    labels: hasDecisionData ? ['Approve', 'Review', 'Reject'] : ['Awaiting prediction'],
    datasets: [
      {
        data: hasDecisionData
          ? [decisionCounts.APPROVE, decisionCounts.REVIEW, decisionCounts.REJECT]
          : [1],
        backgroundColor: hasDecisionData
          ? [DECISION_COLORS.APPROVE, DECISION_COLORS.REVIEW, DECISION_COLORS.REJECT]
          : ['#3b4880'],
        borderColor: '#131620',
        borderWidth: 2,
      },
    ],
  };

  const decisionOptions = {
    ...baseChartOptions,
    plugins: {
      ...baseChartOptions.plugins,
      legend: {
        display: hasDecisionData,
        labels: { color: '#dde3f5', boxWidth: 10, usePointStyle: true },
      },
    },
    cutout: '64%',
  };

  const sourceEntries = Object.entries(docs.source_distribution ?? {}).slice(0, 8);
  const sourceData = {
    labels: sourceEntries.length ? sourceEntries.map(([name]) => name) : ['No source metadata'],
    datasets: [
      {
        label: 'Documents',
        data: sourceEntries.length ? sourceEntries.map(([, count]) => asNumber(count, 0)) : [0],
        backgroundColor: 'rgba(91, 127, 255, 0.75)',
        borderColor: '#7b9aff',
        borderWidth: 1,
        borderRadius: 6,
      },
    ],
  };

  const sourceOptions = {
    ...baseChartOptions,
    indexAxis: 'y',
    plugins: {
      ...baseChartOptions.plugins,
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { color: '#a9b5d7' },
        grid: { color: 'rgba(96, 106, 138, 0.22)' },
      },
      y: {
        ticks: { color: '#a9b5d7' },
        grid: { display: false },
      },
    },
  };

  const lengthBuckets = docs.length_buckets ?? {};
  const lengthData = {
    labels: ['0-100', '101-250', '251-500', '501+'],
    datasets: [
      {
        label: 'Document chunks',
        data: [
          asNumber(lengthBuckets['0-100'], 0),
          asNumber(lengthBuckets['101-250'], 0),
          asNumber(lengthBuckets['251-500'], 0),
          asNumber(lengthBuckets['501+'], 0),
        ],
        borderRadius: 8,
        backgroundColor: [
          'rgba(59, 72, 128, 0.9)',
          'rgba(91, 127, 255, 0.9)',
          'rgba(123, 154, 255, 0.9)',
          'rgba(245, 197, 66, 0.9)',
        ],
      },
    ],
  };

  const lengthOptions = {
    ...baseChartOptions,
    plugins: {
      ...baseChartOptions.plugins,
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { color: '#a9b5d7' },
        grid: { display: false },
      },
      y: {
        ticks: { color: '#a9b5d7' },
        grid: { color: 'rgba(96, 106, 138, 0.22)' },
      },
    },
  };

  const radarData = {
    labels: [
      'Avg PD',
      'Risk Spread',
      'Reject Share',
      'Mean Chunk Words',
      'Retriever Top-K',
      'Threshold Strictness',
    ],
    datasets: [
      {
        label: 'System Signal Score',
        data: [
          Math.min(100, averagePd),
          modelRows.length
            ? Math.min(
                100,
                Math.max(...modelRows.map(row => row.probability)) -
                  Math.min(...modelRows.map(row => row.probability)),
              )
            : 0,
          modelRows.length ? (decisionCounts.REJECT / modelRows.length) * 100 : 0,
          Math.min(100, (asNumber(docs.average_words, 0) / 500) * 100),
          Math.min(100, asNumber(retrieval.top_k, 0) * 10),
          Math.min(100, asNumber(retrieval.score_threshold, 0) * 100),
        ],
        borderColor: '#7b9aff',
        backgroundColor: 'rgba(123, 154, 255, 0.18)',
        pointBackgroundColor: '#dde3f5',
        borderWidth: 2,
      },
    ],
  };

  const radarOptions = {
    ...baseChartOptions,
    plugins: {
      ...baseChartOptions.plugins,
      legend: { display: false },
    },
    scales: {
      r: {
        min: 0,
        max: 100,
        ticks: { display: false, stepSize: 20 },
        angleLines: { color: 'rgba(96, 106, 138, 0.3)' },
        grid: { color: 'rgba(96, 106, 138, 0.3)' },
        pointLabels: { color: '#a9b5d7', font: { size: 11 } },
      },
    },
  };

  return (
    <section className="risk-dashboard-wrap">
      <div className="risk-dashboard-head">
        <div>
          <p className="predict-section-title">📈 Interactive Risk Dashboard</p>
          <p className="risk-dashboard-subtitle">
            Explore submitted borrower inputs, model outcomes, and retrieval corpus health in real time.
          </p>
        </div>
        <button className="dashboard-refresh-btn" onClick={onRefresh} disabled={loading}>
          {loading ? 'Refreshing…' : '↻ Refresh Data'}
        </button>
      </div>

      {error ? <p className="predict-error">⚠️ {error}</p> : null}

      <div className="dashboard-kpi-grid">
        <article className="dashboard-kpi-card">
          <span className="dashboard-kpi-label">Average PD</span>
          <strong className="dashboard-kpi-value">{modelRows.length ? `${averagePd.toFixed(1)}%` : '—'}</strong>
        </article>
        <article className="dashboard-kpi-card">
          <span className="dashboard-kpi-label">Consensus Decision</span>
          <strong className="dashboard-kpi-value">
            {modelRows.length ? `${consensus[0]} (${consensus[1]}/${modelRows.length})` : 'No prediction yet'}
          </strong>
        </article>
        <article className="dashboard-kpi-card">
          <span className="dashboard-kpi-label">Highest Model Risk</span>
          <strong className="dashboard-kpi-value">
            {highestRisk ? `${highestRisk.label} ${highestRisk.probability.toFixed(1)}%` : '—'}
          </strong>
        </article>
        <article className="dashboard-kpi-card">
          <span className="dashboard-kpi-label">Documents Indexed</span>
          <strong className="dashboard-kpi-value">{formatCount(docs.total_documents)}</strong>
        </article>
        <article className="dashboard-kpi-card">
          <span className="dashboard-kpi-label">Observed Requests</span>
          <strong className="dashboard-kpi-value">{formatCount(metrics.window_size)}</strong>
        </article>
        <article className="dashboard-kpi-card">
          <span className="dashboard-kpi-label">P95 Latency</span>
          <strong className="dashboard-kpi-value">{asNumber(metrics.latency_ms?.p95, 0).toFixed(0)} ms</strong>
        </article>
        <article className="dashboard-kpi-card">
          <span className="dashboard-kpi-label">Generation P50</span>
          <strong className="dashboard-kpi-value">{asNumber(metrics.generation_latency_ms?.p50, 0).toFixed(0)} ms</strong>
        </article>
        <article className="dashboard-kpi-card">
          <span className="dashboard-kpi-label">Error Rate</span>
          <strong className="dashboard-kpi-value">{(asNumber(metrics.error_rate, 0) * 100).toFixed(1)}%</strong>
        </article>
      </div>

      <div className="dashboard-chart-grid">
        <article className="dashboard-chart-card dashboard-chart-wide">
          <header className="dashboard-chart-head">
            <h4>Model Prediction Comparison</h4>
            <span>Probability of default by model</span>
          </header>
          <div className="dashboard-chart-box">
            <Bar data={modelProbabilityData} options={modelProbabilityOptions} />
          </div>
        </article>

        <article className="dashboard-chart-card dashboard-chart-wide">
          <header className="dashboard-chart-head">
            <h4>Borrower Input Snapshot</h4>
            <span>Latest submitted application values</span>
          </header>
          {hasBorrowerInput ? (
            <div className="dashboard-input-grid">
              <InputMetric label="Age" value={`${borrowerInput.age.toFixed(0)} years`} />
              <InputMetric label="Annual Income" value={formatCurrency(borrowerInput.income)} />
              <InputMetric label="Loan Amount" value={formatCurrency(borrowerInput.loanAmount)} />
              <InputMetric label="Interest Rate" value={`${borrowerInput.interestRate.toFixed(2)}%`} />
              <InputMetric label="Loan-to-Income" value={borrowerInput.loanToIncome === null ? 'N/A' : `${borrowerInput.loanToIncome.toFixed(1)}%`} />
              <InputMetric label="Employment" value={`${borrowerInput.employmentLength.toFixed(1)} years`} />
              <InputMetric label="Credit History" value={`${borrowerInput.creditHistoryLength.toFixed(1)} years`} />
              <InputMetric label="Home Ownership" value={borrowerInput.homeOwnership} />
              <InputMetric label="Loan Purpose" value={formatIntent(borrowerInput.loanIntent)} />
              <InputMetric label="Prior Default" value={borrowerInput.priorDefault} />
            </div>
          ) : (
            <p className="dashboard-empty">Run a prediction to display borrower inputs alongside model results.</p>
          )}
        </article>

        <article className="dashboard-chart-card dashboard-chart-wide">
          <header className="dashboard-chart-head">
            <h4>Input vs Result Risk View</h4>
            <span>Normalized borrower pressure factors with average model PD overlay</span>
          </header>
          <div className="dashboard-chart-box">
            <Bar data={inputRiskData} options={inputRiskOptions} />
          </div>
        </article>

        <article className="dashboard-chart-card">
          <header className="dashboard-chart-head">
            <h4>Decision Split</h4>
            <span>Approve vs Review vs Reject</span>
          </header>
          <div className="dashboard-chart-box">
            <Doughnut data={decisionData} options={decisionOptions} />
          </div>
        </article>

        <article className="dashboard-chart-card">
          <header className="dashboard-chart-head">
            <h4>Risk and Retrieval Signals</h4>
            <span>Normalized health view (0-100)</span>
          </header>
          <div className="dashboard-chart-box">
            <Radar data={radarData} options={radarOptions} />
          </div>
        </article>

        <article className="dashboard-chart-card dashboard-chart-wide">
          <header className="dashboard-chart-head">
            <h4>Document Source Distribution</h4>
            <span>
              {docs.sampled
                ? `Top sources from ${formatCount(docs.sample_size)} sampled chunks`
                : `All ${formatCount(docs.sample_size)} chunks in corpus`}
            </span>
          </header>
          <div className="dashboard-chart-box">
            <Bar data={sourceData} options={sourceOptions} />
          </div>
        </article>

        <article className="dashboard-chart-card dashboard-chart-wide">
          <header className="dashboard-chart-head">
            <h4>Chunk Length Buckets</h4>
            <span>
              Avg words {asNumber(docs.average_words, 0).toFixed(1)} · Median words{' '}
              {asNumber(docs.median_words, 0).toFixed(1)}
            </span>
          </header>
          <div className="dashboard-chart-box">
            <Bar data={lengthData} options={lengthOptions} />
          </div>
        </article>
      </div>
    </section>
  );
}
