import React, { useState, useRef, useMemo } from 'react'
import './app.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:3001'

const PRESETS = {
  Small: { size: 1000, dim: 64, out: 'data/test_small' },
  Medium: { size: 10000, dim: 128, out: 'data/test_medium' },
  Large: { size: 100000, dim: 256, out: 'data/test_large' },
}

function cx(...xs) {
  return xs.filter(Boolean).join(' ')
}

function StatusBadge({ status }) {
  if (!status) return <span className="badge badge-muted">idle</span>
  const s = status.toLowerCase()
  if (s === 'running') return <span className="badge badge-info">running</span>
  if (s === 'completed') return <span className="badge badge-ok">completed</span>
  if (s === 'failed' || s === 'error') return <span className="badge badge-bad">{s}</span>
  return <span className="badge badge-muted">{s}</span>
}

export default function App() {
  const [size, setSize] = useState(PRESETS.Medium.size)
  const [dim, setDim] = useState(PRESETS.Medium.dim)
  const [out, setOut] = useState(PRESETS.Medium.out)
  const [dataDir, setDataDir] = useState(PRESETS.Medium.out)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [stdout, setStdout] = useState('')
  const [stderr, setStderr] = useState('')
  const [plots, setPlots] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [activeLog, setActiveLog] = useState('stdout')
  const logRef = useRef(null)

  const isRunning = jobStatus?.status === 'running'

  function applyPreset(name) {
    const p = PRESETS[name]
    if (!p) return
    setSize(p.size)
    setDim(p.dim)
    setOut(p.out)
    setDataDir(p.out)
  }

  async function startGenerate() {
    const resp = await fetch(`${API}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ size, dim, out }),
    })
    const j = await resp.json()
    setJobId(j.jobId)
    setStdout('')
    setStderr('')
    pollJob(j.jobId)
  }

  async function startIngest(target) {
    const resp = await fetch(`${API}/ingest/${target}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data_dir: dataDir }),
    })
    const j = await resp.json()
    setJobId(j.jobId)
    setStdout('')
    setStderr('')
    pollJob(j.jobId)
  }

  async function docker(action) {
    const resp = await fetch(`${API}/docker/${action}`, { method: 'POST' })
    const j = await resp.json()
    setJobId(j.jobId)
    setStdout('')
    setStderr('')
    pollJob(j.jobId)
  }

  function pollJob(id) {
    setJobStatus({ status: 'running' })
    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`${API}/jobs/${id}`)
        if (!resp.ok) return
        const s = await resp.json()
        setJobStatus(s)
        if (s.stdout != null) setStdout(s.stdout)
        if (s.stderr != null) setStderr(s.stderr)
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
        if (s.status === 'completed' || s.status === 'failed' || s.status === 'error') {
          clearInterval(interval)
        }
      } catch {}
    }, 1000)
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text || '')
    } catch {}
  }

  const progress = useMemo(() => {
    if (!jobStatus) return null
    const s = jobStatus.status
    if (s === 'running') return <div className="bar bar-anim" />
    if (s === 'completed') return <div className="bar bar-full" />
    if (s === 'failed' || s === 'error') return <div className="bar bar-bad" />
    return null
  }, [jobStatus])

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="kicker">Vec3</div>
          <h1 className="title">Runner</h1>
        </div>
        <div className="headerRight">
          <div className="meta">
            <div className="metaRow">
              <span className="metaLabel">API</span>
              <span className="mono metaValue">{API}</span>
            </div>
            <div className="metaRow">
              <span className="metaLabel">Job</span>
              <span className="mono metaValue">{jobId || '—'}</span>
            </div>
          </div>
          <div className="status">
            <StatusBadge status={jobStatus?.status} />
          </div>
        </div>
      </header>

      <div className="grid">
        <section className="card">
          <div className="cardHeader">
            <h3>Docker</h3>
            <span className="subtle">Manage services</span>
          </div>
          <div className="row">
            <button className="btn btnPrimary" onClick={() => docker('up')} disabled={isRunning}>
              Docker Up
            </button>
            <button className="btn" onClick={() => docker('down')} disabled={isRunning}>
              Docker Down
            </button>
          </div>
        </section>

        <section className="card span2">
          <div className="cardHeader">
            <h3>Results & Plots</h3>
            <span className="subtle">View generated charts and metrics</span>
          </div>

          <div className="row">
            <button
              className="btn"
              onClick={async () => {
                try {
                  const r = await fetch(`${API}/plots`)
                  const j = await r.json()
                  setPlots(j.plots || [])
                } catch (err) {
                  setPlots([])
                }
              }}
            >
              Refresh Plots
            </button>
            <button
              className="btn"
              onClick={async () => {
                try {
                  const r = await fetch(`${API}/metrics`)
                  if (!r.ok) {
                    alert('No metrics found')
                    return
                  }
                  const m = await r.json()
                  setMetrics(m)
                } catch (err) {
                  alert('Failed to load metrics')
                }
              }}
            >
              Load Metrics
            </button>
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{ display: 'flex', gap: 12 }}>
              <div style={{ flex: 1 }}>
                <h4>Plots</h4>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {plots && plots.length ? (
                    plots.map((p) => (
                      <div key={p} style={{ width: 200 }}>
                        <img src={`${API}/plots/${p}`} alt={p} style={{ width: '100%' }} />
                        <div style={{ fontSize: 12, textAlign: 'center' }}>{p}</div>
                      </div>
                    ))
                  ) : (
                    <div>(no plots)</div>
                  )}
                </div>
              </div>

              <div style={{ flex: 1 }}>
                <h4>Metrics</h4>
                <pre style={{ height: 300, overflow: 'auto', background: '#f6f8fa', padding: 8 }}>
                  {metrics ? JSON.stringify(metrics, null, 2) : '(no metrics loaded)'}
                </pre>
              </div>
            </div>
          </div>
        </section>

        <section className="card">
          <div className="cardHeader">
            <h3>Presets</h3>
            <span className="subtle">Quick sizes</span>
          </div>
          <div className="row wrap">
            {Object.keys(PRESETS).map((k) => (
              <button key={k} className="pill" onClick={() => applyPreset(k)} disabled={isRunning}>
                {k}
              </button>
            ))}
          </div>
        </section>

        <section className="card span2">
          <div className="cardHeader">
            <h3>Generate dataset</h3>
            <span className="subtle">Create vectors on disk</span>
          </div>

          <div className="form">
            <label className="field">
              <span>Size</span>
              <input
                type="number"
                value={size}
                onChange={(e) => setSize(Number(e.target.value))}
                disabled={isRunning}
              />
            </label>

            <label className="field">
              <span>Dim</span>
              <input
                type="number"
                value={dim}
                onChange={(e) => setDim(Number(e.target.value))}
                disabled={isRunning}
              />
            </label>

            <label className="field fieldWide">
              <span>Out</span>
              <input value={out} onChange={(e) => setOut(e.target.value)} disabled={isRunning} />
            </label>
          </div>

          <div className="row">
            <button className="btn btnPrimary" onClick={startGenerate} disabled={isRunning}>
              Generate
            </button>
            <div className="progressWrap" aria-hidden="true">
              {progress}
            </div>
          </div>
        </section>

        <section className="card span2">
          <div className="cardHeader">
            <h3>Ingest</h3>
            <span className="subtle">Load dataset into a target</span>
          </div>

          <div className="form">
            <label className="field fieldWide">
              <span>Data dir</span>
              <input value={dataDir} onChange={(e) => setDataDir(e.target.value)} disabled={isRunning} />
            </label>
          </div>

          <div className="row">
            <button className="btn btnPrimary" onClick={() => startIngest('chroma')} disabled={isRunning}>
              Ingest → Chroma
            </button>
            <button className="btn" onClick={() => startIngest('pgvector')} disabled={isRunning}>
              Ingest → pgvector
            </button>
            <div className="progressWrap" aria-hidden="true">
              {progress}
            </div>
          </div>
        </section>

        <section className="card span2">
          <div className="cardHeader">
            <h3>Logs</h3>
            <span className="subtle">Live output from current job</span>
          </div>

          <div className="logShell">
            <div className="logTop">
              <div className="tabs">
                <button
                  className={cx('tab', activeLog === 'stdout' && 'tabActive')}
                  onClick={() => setActiveLog('stdout')}
                >
                  Stdout
                </button>
                <button
                  className={cx('tab', activeLog === 'stderr' && 'tabActive')}
                  onClick={() => setActiveLog('stderr')}
                >
                  Stderr
                </button>
              </div>
              <div className="row">
                <button className="btn btnSmall" onClick={() => copy(activeLog === 'stdout' ? stdout : stderr)}>
                  Copy
                </button>
                <button
                  className="btn btnSmall"
                  onClick={() => (activeLog === 'stdout' ? setStdout('') : setStderr(''))}
                >
                  Clear
                </button>
              </div>
            </div>

            <div
              ref={logRef}
              className={cx('logBody', activeLog === 'stderr' && 'logErr')}
            >
              <pre className="mono pre">
                {activeLog === 'stdout'
                  ? stdout || '(no stdout yet)'
                  : stderr || '(no stderr yet)'}
              </pre>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
