import '../style/Sidebar.css';
const PROMPTS = [
  'What is credit risk?',
  'What factors affect a credit score?',
  'Why was my loan application rejected?',
  'How can I improve my chances of getting a loan?',
  'What is a debt-to-income ratio?',
  'What does loan grade mean?',
  'How does employment length affect loan approval?',
  'What is the difference between secured and unsecured loans?',
  'How does interest rate relate to credit risk?',
  'What is a non-performing loan?',
];

const MODEL_TITLES = {
  'llama-1b': 'Llama 3.2 1B Instruct',
  'llama-3b': 'Llama 3.2 3B Instruct',
};

const MODEL_BADGES = {
  'llama-1b': 'Llama 3.2 - 1B',
  'llama-3b': 'Llama 3.2 - 3B',
};

export default function Sidebar({ onPromptClick, onClear, hasMsgs, activeModel = 'llama-1b' }) {
  const modelTitle = MODEL_TITLES[activeModel] ?? MODEL_TITLES['llama-1b'];
  const modelBadge = MODEL_BADGES[activeModel] ?? MODEL_BADGES['llama-1b'];

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span>AI</span>
        <strong>{modelTitle}</strong>
      </div>

      <section className="sidebar-section">
        <p className="sidebar-label">FAQ</p>
        {PROMPTS.map(p => (
          <button key={p} className="prompt-btn" onClick={() => onPromptClick(p)}>{p}</button>
        ))}
      </section>

      {hasMsgs && (
        <button className="clear-btn" onClick={onClear}>Clear chat</button>
      )}

      <div className="sidebar-footer">
        <span>Credit Risk Assistant</span>
        <span className="badge">{modelBadge}</span>
      </div>
    </aside>
  );
}
