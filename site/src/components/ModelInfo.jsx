export default function ModelInfo({ model }) {
  return (
    <div className="model-info">
      <div className="panel-title">Model Details</div>

      <p className="info-desc">{model.desc}</p>

      <div className="info-tags">
        {model.tags.map(tag => (
          <span key={tag} className="tag">{tag}</span>
        ))}
      </div>

      <div className="info-grid">
        <div className="info-item">
          <span className="info-label">Steps</span>
          <span className="info-value">{model.steps}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Training</span>
          <span className="info-value">{model.time}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Avg Dist</span>
          <span className="info-value">{model.avgDist.toLocaleString()}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Clear Rate</span>
          <span className="info-value" style={{ color: model.clearRate > 0 ? model.color : '#555' }}>
            {model.clearRate}%
          </span>
        </div>
      </div>
    </div>
  )
}
