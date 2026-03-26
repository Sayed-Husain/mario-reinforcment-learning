import { useEffect } from 'react'

const OBSTACLES = [
  { pos: 450, label: 'Pipes' },
  { pos: 1300, label: 'Gap' },
  { pos: 2400, label: 'Stairs' },
]

export default function GameScreen({ videoRef, src, playState, cleared, lastFrame, totalFrames, levelLength, distance, onRun, onEnded, speed, onSpeedChange }) {
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    video.addEventListener('ended', onEnded)
    return () => video.removeEventListener('ended', onEnded)
  }, [videoRef, onEnded])

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed
  }, [speed, videoRef])

  const distPct = Math.min((distance / levelLength) * 100, 100)

  return (
    <div className="game-area">
      <div className="screen-frame">
        <video ref={videoRef} src={src} muted playsInline preload="auto" />
        <div className="scanlines" />

        <div className={`screen-overlay ${playState !== 'idle' ? 'hidden' : ''}`}>
          <button className="run-btn" onClick={onRun}>&#9654; RUN MODEL</button>
        </div>

        <div className={`screen-overlay ${playState !== 'ended' ? 'hidden' : ''}`}>
          <div className={`end-label ${cleared ? 'cleared' : ''}`}>
            {cleared ? 'LEVEL CLEARED' : 'GAME OVER'}
          </div>
          <div className="end-stats">
            {lastFrame && (
              <>
                Distance: {lastFrame.distance.toLocaleString()} / {levelLength.toLocaleString()}<br />
                Reward: {Math.round(lastFrame.reward).toLocaleString()}<br />
                Frames: {totalFrames.toLocaleString()}
              </>
            )}
          </div>
          <button className="run-btn" onClick={onRun}>&#9654; RUN AGAIN</button>
        </div>
      </div>

      {/* Level progress with obstacle markers */}
      <div className="level-progress">
        <div className="level-bar">
          <div
            className={`level-fill ${distance >= levelLength ? 'cleared' : ''}`}
            style={{ width: `${distPct}%` }}
          />
          <div className="level-marker" style={{ left: `${distPct}%` }} />
          {OBSTACLES.map(o => (
            <div
              key={o.pos}
              className="obstacle-tick"
              style={{ left: `${(o.pos / levelLength) * 100}%` }}
              title={o.label}
            />
          ))}
        </div>
        <div className="level-labels">
          <span>Start</span>
          {OBSTACLES.map(o => (
            <span
              key={o.pos}
              className="obstacle-label"
              style={{ left: `${(o.pos / levelLength) * 100}%` }}
            >
              {o.label}
            </span>
          ))}
          <span>Flag</span>
        </div>
      </div>

      {/* Speed control */}
      <div className="speed-control">
        {[0.5, 1, 2].map(s => (
          <button
            key={s}
            className={`speed-btn ${speed === s ? 'active' : ''}`}
            onClick={() => onSpeedChange(s)}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  )
}
