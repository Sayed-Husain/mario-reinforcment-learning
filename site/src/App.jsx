import { useState, useRef, useEffect, useCallback } from 'react'
import GameScreen from './components/GameScreen'
import StatsPanel from './components/StatsPanel'
import ModelCard from './components/ModelCard'
import ModelInfo from './components/ModelInfo'
import Pipeline from './components/Pipeline'
import CompareView from './components/CompareView'

const BASE = import.meta.env.BASE_URL

const MODELS = {
  rainbow: {
    label: 'Rainbow DQN',
    video: `${BASE}videos/rainbow.mp4`,
    data: `${BASE}data/rainbow.json`,
    color: '#e7413c',
    clearRate: 45,
    avgDist: 2602,
    steps: '2M',
    time: '~13h',
    year: '2017',
    desc: 'Double DQN + prioritized experience replay + dueling network architecture',
    tags: ['Custom', 'Double DQN', 'Prioritized Replay', 'Dueling Net'],
  },
  dqn: {
    label: 'DQN',
    video: `${BASE}videos/dqn.mp4`,
    data: `${BASE}data/dqn.json`,
    color: '#6b8cff',
    clearRate: 35,
    avgDist: 2518,
    steps: '2M',
    time: '~16h',
    year: '2015',
    desc: 'Custom Double DQN with experience replay and epsilon-greedy exploration',
    tags: ['Custom', 'Double DQN', 'Replay Buffer'],
  },
  ppo: {
    label: 'PPO',
    video: `${BASE}videos/ppo.mp4`,
    data: `${BASE}data/ppo.json`,
    color: '#ffa726',
    clearRate: 0,
    avgDist: 2026,
    steps: '2M',
    time: '~5h',
    year: '2017',
    desc: 'Proximal Policy Optimization via Stable-Baselines3 with CNN policy',
    tags: ['Stable-Baselines3', 'Policy Gradient', 'Parallel Envs'],
  },
}

const MODEL_KEYS = Object.keys(MODELS)
const LEVEL_LENGTH = 3161

export default function App() {
  const [mode, setMode] = useState('single') // 'single' or 'compare'
  const [activeModel, setActiveModel] = useState('rainbow')
  const [frameData, setFrameData] = useState([])
  const [stats, setStats] = useState({ distance: 0, reward: 0, action: '\u2014', frame: 0 })
  const [playState, setPlayState] = useState('idle')
  const [speed, setSpeed] = useState(1)
  const videoRef = useRef(null)
  const animRef = useRef(null)

  const model = MODELS[activeModel]

  useEffect(() => {
    if (mode !== 'single') return
    setPlayState('idle')
    setStats({ distance: 0, reward: 0, action: '\u2014', frame: 0 })
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.currentTime = 0
    }
    fetch(model.data)
      .then(r => r.json())
      .then(setFrameData)
      .catch(() => setFrameData([]))
  }, [model.data, mode])

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

  const handleRun = () => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = 0
    video.play()
    setPlayState('playing')
    animRef.current = requestAnimationFrame(updateStats)
  }

  const handleEnded = () => {
    setPlayState('ended')
    cancelAnimationFrame(animRef.current)
  }

  const handleSwitch = (key) => {
    cancelAnimationFrame(animRef.current)
    setActiveModel(key)
  }

  const handleModeToggle = (newMode) => {
    cancelAnimationFrame(animRef.current)
    setMode(newMode)
  }

  const lastFrame = frameData[frameData.length - 1]
  const cleared = lastFrame && lastFrame.distance >= LEVEL_LENGTH

  return (
    <div className="dashboard">
      <header className="header">
        <div className="header-left">
          <span className="logo">MARIO RL</span>
          <span className="header-sep">/</span>
          <span className="header-level">World 1-1</span>
        </div>
        <div className="header-right">
          <div className="mode-toggle">
            <button
              className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
              onClick={() => handleModeToggle('single')}
            >
              Single
            </button>
            <button
              className={`mode-btn ${mode === 'compare' ? 'active' : ''}`}
              onClick={() => handleModeToggle('compare')}
            >
              Compare
            </button>
          </div>
          <a className="github-link" href="https://github.com/Sayed-Husain/mario-reinforcment-learning" target="_blank" rel="noreferrer">
            GitHub
          </a>
        </div>
      </header>

      <div className={`main ${mode === 'compare' ? 'compare-mode' : ''}`}>
        {/* Left panel */}
        <div className="panel-left">
          {mode === 'single' ? (
            <>
              <div className="panel-title">Select Model</div>
              {Object.entries(MODELS).map(([key, m]) => (
                <ModelCard
                  key={key}
                  model={m}
                  active={key === activeModel}
                  onClick={() => handleSwitch(key)}
                />
              ))}
              <ModelInfo model={model} />
            </>
          ) : (
            <>
              <div className="panel-title">Comparing All Models</div>
              {Object.entries(MODELS).map(([key, m]) => (
                <div key={key} className="compare-legend-item">
                  <span className="compare-legend-dot" style={{ background: m.color }} />
                  <span className="compare-legend-name">{m.label}</span>
                  <span className="compare-legend-rate">{m.clearRate}%</span>
                </div>
              ))}
              <div className="speed-control-sidebar">
                <div className="panel-title">Speed</div>
                <div className="speed-control">
                  {[0.5, 1, 2].map(s => (
                    <button
                      key={s}
                      className={`speed-btn ${speed === s ? 'active' : ''}`}
                      onClick={() => setSpeed(s)}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>
              <ModelInfo model={MODELS.rainbow} />
            </>
          )}
        </div>

        {/* Center */}
        <div className="panel-center">
          {mode === 'single' ? (
            <>
              <GameScreen
                videoRef={videoRef}
                src={model.video}
                playState={playState}
                cleared={cleared}
                lastFrame={lastFrame}
                totalFrames={frameData.length}
                levelLength={LEVEL_LENGTH}
                distance={stats.distance}
                onRun={handleRun}
                onEnded={handleEnded}
                speed={speed}
                onSpeedChange={setSpeed}
              />
              <Pipeline />
            </>
          ) : (
            <CompareView
              models={MODELS}
              compareModels={MODEL_KEYS}
              speed={speed}
            />
          )}
        </div>

        {/* Right panel  - only in single mode */}
        {mode === 'single' && (
          <div className="panel-right">
            <StatsPanel
              model={model}
              playState={playState}
              stats={stats}
              totalFrames={frameData.length}
              levelLength={LEVEL_LENGTH}
              activeModel={activeModel}
            />
          </div>
        )}
      </div>
    </div>
  )
}
