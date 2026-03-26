export default function ModelCard({ model, active, onClick }) {
  return (
    <button
      className={`model-card ${active ? 'active' : ''}`}
      onClick={onClick}
      style={{ '--card-color': model.color }}
    >
      <div className="card-header">
        <span className="card-name">{model.label}</span>
        <span className="card-rate" style={{ color: model.clearRate > 0 ? model.color : '#555' }}>
          {model.clearRate}%
        </span>
      </div>
      <div className="card-bar-wrap">
        <div
          className="card-bar"
          style={{ width: `${Math.max(model.clearRate, 2)}%`, background: model.color }}
        />
      </div>
      <div className="card-meta">
        <span>{model.steps} steps</span>
        <span>{model.time}</span>
      </div>
    </button>
  )
}
