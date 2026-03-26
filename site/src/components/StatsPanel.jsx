const COMPARE_DATA = [
  { key: 'rainbow', label: 'Rainbow', dist: 2602, rate: 45, color: '#e7413c' },
  { key: 'dqn', label: 'DQN', dist: 2518, rate: 35, color: '#6b8cff' },
  { key: 'ppo', label: 'PPO', dist: 2026, rate: 0, color: '#ffa726' },
]

export default function StatsPanel({ model, playState, stats, totalFrames, levelLength, activeModel }) {
  const distPct = Math.min((stats.distance / levelLength) * 100, 100)
  const isCleared = stats.distance >= levelLength

  const statusDot = playState === 'playing' ? 'running' : playState === 'ended' ? 'done' : ''
  const statusText = playState === 'playing' ? 'Inference running' : playState === 'ended' ? 'Complete' : 'Awaiting input'

  return (
    <div className="stats-panel">
      <div className="panel-title">Live Output</div>

      <div className="status-row">
        <span className={`dot ${statusDot}`} />
        <span className="status-text">{statusText}</span>
      </div>

      <div className="stat-group">
        <div className="stat-block">
          <span className="stat-label">Distance</span>
          <span className="stat-number">{stats.distance.toLocaleString()}</span>
          <div className="stat-bar-wrap">
            <div
              className={`stat-bar-fill ${isCleared ? 'cleared' : ''}`}
              style={{ width: `${distPct}%` }}
            />
          </div>
        </div>

        <div className="stat-block">
          <span className="stat-label">Reward</span>
          <span className="stat-number">{stats.reward.toLocaleString()}</span>
        </div>

        <div className="stat-block">
          <span className="stat-label">Action</span>
          <span className="stat-action">{stats.action}</span>
        </div>

        <div className="stat-block">
          <span className="stat-label">Frame</span>
          <span className="stat-number">{stats.frame.toLocaleString()} <span className="stat-dim">/ {totalFrames.toLocaleString()}</span></span>
        </div>
      </div>

      {/* Comparison  - always visible */}
      <div className="compare-section">
        <div className="panel-title">Avg Distance</div>
        {COMPARE_DATA.map(c => (
          <div className={`compare-row ${c.key === activeModel ? 'highlight' : ''}`} key={c.key}>
            <span className="compare-name" style={{ color: c.key === activeModel ? c.color : undefined }}>{c.label}</span>
            <div className="compare-bar-wrap">
              <div
                className="compare-bar"
                style={{
                  width: `${(c.dist / levelLength) * 100}%`,
                  background: c.color,
                  opacity: c.key === activeModel ? 1 : 0.35,
                }}
              />
            </div>
            <span className="compare-val">{c.dist.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
