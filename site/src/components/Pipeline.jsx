const STEPS = [
  { name: 'Raw Input', detail: '240\u00d7256 RGB', icon: '\ud83c\udfae' },
  { name: 'Grayscale', detail: 'Drop color channels', icon: '\u25a8' },
  { name: 'Resize', detail: '84\u00d784 pixels', icon: '\u2b1c' },
  { name: 'Stack', detail: '4 frames \u2192 motion', icon: '\u25a6' },
  { name: 'CNN', detail: '3 conv + 2 fc layers', icon: '\ud83e\udde0' },
  { name: 'Action', detail: '7 possible moves', icon: '\u2192' },
]

export default function Pipeline() {
  return (
    <div className="pipeline">
      <div className="pipeline-label">Observation Pipeline</div>
      <div className="pipeline-flow">
        {STEPS.map((step, i) => (
          <div key={step.name} className="pipeline-node-wrap">
            <div className="pipeline-node">
              <span className="pipeline-icon">{step.icon}</span>
              <span className="pipeline-name">{step.name}</span>
              <span className="pipeline-detail">{step.detail}</span>
            </div>
            {i < STEPS.length - 1 && <div className="pipeline-connector" />}
          </div>
        ))}
      </div>
    </div>
  )
}
