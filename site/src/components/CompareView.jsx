import { useRef, useEffect, useState, useCallback } from 'react'

const LEVEL_LENGTH = 3161

function ComparePlayer({ modelKey, model, speed, runSignal }) {
  const videoRef = useRef(null)
  const animRef = useRef(null)
  const [frameData, setFrameData] = useState([])
  const [stats, setStats] = useState({ distance: 0, reward: 0, action: '\u2014', frame: 0 })
  const [playState, setPlayState] = useState('idle')

  useEffect(() => {
    fetch(model.data).then(r => r.json()).then(setFrameData).catch(() => setFrameData([]))
  }, [model.data])

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed
  }, [speed])

  const updateStats = useCallback(() => {
    const video = videoRef.current
    if (!video || video.paused || video.ended || frameData.length === 0) return
    const progress = video.currentTime / video.duration
    const idx = Math.min(Math.floor(progress * frameData.length), frameData.length - 1)
    const d = frameData[idx]
    if (d) {
      setStats({ distance: d.distance, reward: Math.round(d.reward), action: d.action, frame: idx + 1 })
    }
    animRef.current = requestAnimationFrame(updateStats)
  }, [frameData])

  // Listen for run signal from parent
  useEffect(() => {
    if (runSignal === 0) return
    const video = videoRef.current
    if (!video) return
    video.currentTime = 0
    video.play()
    setPlayState('playing')
    setStats({ distance: 0, reward: 0, action: '\u2014', frame: 0 })
    animRef.current = requestAnimationFrame(updateStats)
  }, [runSignal, updateStats])

  const handleEnded = () => {
    setPlayState('ended')
    cancelAnimationFrame(animRef.current)
  }

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    video.addEventListener('ended', handleEnded)
    return () => video.removeEventListener('ended', handleEnded)
  }, [])

  const lastFrame = frameData[frameData.length - 1]
  const cleared = lastFrame && lastFrame.distance >= LEVEL_LENGTH
  const distPct = Math.min((stats.distance / LEVEL_LENGTH) * 100, 100)

  return (
    <div className="compare-player">
      <div className="compare-label" style={{ color: model.color }}>{model.label}</div>
      <div className="compare-screen-frame">
        <video ref={videoRef} src={model.video} muted playsInline preload="auto" />
        <div className="scanlines" />
        {playState === 'ended' && (
          <div className="compare-end-badge" style={{ background: cleared ? 'var(--green)' : 'var(--accent)' }}>
            {cleared ? 'CLEARED' : 'GAME OVER'}
          </div>
        )}
      </div>
      <div className="compare-stats-row">
        <div className="compare-stat">
          <span className="compare-stat-label">Distance</span>
          <span className="compare-stat-val">{stats.distance.toLocaleString()}</span>
        </div>
        <div className="compare-stat">
          <span className="compare-stat-label">Reward</span>
          <span className="compare-stat-val">{stats.reward.toLocaleString()}</span>
        </div>
      </div>
      <div className="compare-progress">
        <div className="compare-progress-fill" style={{ width: `${distPct}%`, background: model.color }} />
      </div>
    </div>
  )
}

export default function CompareView({ models, compareModels, speed }) {
  const [runSignal, setRunSignal] = useState(0)

  const handleRunAll = () => setRunSignal(s => s + 1)

  return (
    <div className="compare-view">
      <div className="compare-grid">
        {compareModels.map(key => (
          <ComparePlayer
            key={key}
            modelKey={key}
            model={models[key]}
            speed={speed}
            runSignal={runSignal}
          />
        ))}
      </div>
      <div className="compare-controls">
        <button className="run-btn" onClick={handleRunAll}>&#9654; RUN ALL</button>
      </div>
    </div>
  )
}
