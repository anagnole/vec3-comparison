import React, { useState, useRef, useMemo, useEffect } from 'react'
import './app.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:3001'


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

function ContainerBadge({ status }) {
  if (!status) return <span className="badge badge-muted">unknown</span>
  const s = status.toLowerCase()
  if (s === 'running') return <span className="badge badge-ok">running</span>
  if (s === 'exited') return <span className="badge badge-bad">stopped</span>
  return <span className="badge badge-muted">{s}</span>
}


export default function App() {
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [stdout, setStdout] = useState('')
  const [stderr, setStderr] = useState('')
  const [activeLog, setActiveLog] = useState('stdout')
  const logRef = useRef(null)
  const isRunning = jobStatus?.status === 'running'

  const [dockerStatus, setDockerStatus] = useState({ chroma: null, pgvector: null })

  const [datasets, setDatasets] = useState([])

  const [genSize, setGenSize] = useState(10000)
  const [genDim, setGenDim] = useState(128)
  const [genOut, setGenOut] = useState('data/10k')
  const [genDist, setGenDist] = useState('uniform')
  const [genClasses, setGenClasses] = useState('A,B,C')
  const [genClassDist, setGenClassDist] = useState('')
  const [genSeed, setGenSeed] = useState('')

  const [ingestDataset, setIngestDataset] = useState('')
  const [ingestBatchSize, setIngestBatchSize] = useState(1000)
  const [ingestIndexType, setIngestIndexType] = useState('ivfflat')
  const [ingestLists, setIngestLists] = useState(100)
  const [ingestHnswM, setIngestHnswM] = useState(16)
  const [ingestHnswEf, setIngestHnswEf] = useState(64)
  const [ingestFresh, setIngestFresh] = useState(false)

  const [queryDataset, setQueryDataset] = useState('')
  const [queryMode, setQueryMode] = useState('both')
  const [queryIndexType, setQueryIndexType] = useState('ivfflat')
  const [queryMetric, setQueryMetric] = useState('euclidean')
  const [queryNoRestart, setQueryNoRestart] = useState(false)
  const [queryFresh, setQueryFresh] = useState(false)

  const [ingestionPlots, setIngestionPlots] = useState([])
  const [queryPlots, setQueryPlots] = useState([])

  useEffect(() => {
    loadDockerStatus()
    loadDatasets()
    loadPlots()
  }, [])

  async function loadDockerStatus() {
    try {
      const r = await fetch(`${API}/docker/status`)
      const j = await r.json()
      setDockerStatus(j)
    } catch {}
  }

  async function loadDatasets() {
    try {
      const r = await fetch(`${API}/datasets`)
      const j = await r.json()
      setDatasets(j.datasets || [])
      if (j.datasets?.length && !ingestDataset) {
        setIngestDataset(j.datasets[0].name)
        setQueryDataset(j.datasets[0].name)
      }
    } catch {}
  }

  async function loadPlots() {
    try {
      const r = await fetch(`${API}/plots/ingestion`)
      const j = await r.json()
      // Filter to only existing plots, extract just the filename
      const existing = (j.plots || []).filter(p => p.exists).map(p => `${p.name}.png`)
      setIngestionPlots(existing)
    } catch {}
    try {
      const r = await fetch(`${API}/plots/queries`)
      const j = await r.json()
      const existing = (j.plots || []).filter(p => p.exists).map(p => `${p.name}.png`)
      setQueryPlots(existing)
    } catch {}
  }

  function pollJob(id, onComplete) {
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
          if (onComplete) onComplete(s.status === 'completed')
        }
      } catch {}
    }, 1000)
  }

  async function docker(action) {
    setStdout('')
    setStderr('')
    const resp = await fetch(`${API}/docker/${action}`, { method: 'POST' })
    const j = await resp.json()
    setJobId(j.jobId)
    pollJob(j.jobId, () => loadDockerStatus())
  }

  async function startGenerate() {
    setStdout('')
    setStderr('')
    const body = {
      size: genSize,
      dim: genDim,
      out: genOut,
      distribution: genDist,
    }
    if (genClasses) body.classes = genClasses
    if (genClassDist) body.classDist = genClassDist
    if (genSeed) body.seed = parseInt(genSeed, 10)
    
    const resp = await fetch(`${API}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const j = await resp.json()
    setJobId(j.jobId)
    pollJob(j.jobId, () => loadDatasets())
  }

  async function startIngestion() {
    setStdout('')
    setStderr('')
    const body = {
      dataset: ingestDataset,
      batchSize: ingestBatchSize,
      indexType: ingestIndexType,
      lists: ingestLists,
      hnswM: ingestHnswM,
      hnswEf: ingestHnswEf,
      fresh: ingestFresh,
    }
    const resp = await fetch(`${API}/benchmark/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const j = await resp.json()
    setJobId(j.jobId)
    pollJob(j.jobId)
  }

  async function startQuery() {
    setStdout('')
    setStderr('')
    const body = {
      dataset: queryDataset,
      mode: queryMode,
      indexType: queryIndexType,
      metric: queryMetric,
      noRestart: queryNoRestart,
      fresh: queryFresh,
    }
    const resp = await fetch(`${API}/benchmark/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const j = await resp.json()
    setJobId(j.jobId)
    pollJob(j.jobId)
  }

  async function generatePlot(type, plotName) {
    setStdout('')
    setStderr('')
    const resp = await fetch(`${API}/plots/${type}/${plotName}`, { method: 'POST' })
    const j = await resp.json()
    setJobId(j.jobId)
    pollJob(j.jobId, () => loadPlots())
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
          <h1 className="title">Benchmark Runner</h1>
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
            <span className="subtle">Manage database containers</span>
          </div>
          <div className="row" style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
              <span>Chroma: <ContainerBadge status={dockerStatus.chroma} /></span>
              <span>pgvector: <ContainerBadge status={dockerStatus.pgvector} /></span>
              <button className="btn btnSmall" onClick={loadDockerStatus}>Refresh</button>
            </div>
          </div>
          <div className="row">
            <button className="btn btnPrimary" onClick={() => docker('up')} disabled={isRunning}>
              Start
            </button>
            <button className="btn" onClick={() => docker('down')} disabled={isRunning}>
              Stop
            </button>
            <button className="btn" onClick={() => docker('restart')} disabled={isRunning}>
              Restart
            </button>
          </div>
        </section>

        <section className="card span2">
          <div className="cardHeader">
            <h3>Data Generator</h3>
            <span className="subtle">Create vector datasets</span>
          </div>

          <div style={{ display: 'flex', gap: 24 }}>
            <div style={{ flex: 1 }}>
              <div className="form">
                <label className="field">
                  <span>Size</span>
                  <input type="number" value={genSize} onChange={(e) => setGenSize(Number(e.target.value))} disabled={isRunning} />
                </label>
                <label className="field">
                  <span>Dimensions</span>
                  <input type="number" value={genDim} onChange={(e) => setGenDim(Number(e.target.value))} disabled={isRunning} />
                </label>
                <label className="field fieldWide">
                  <span>Output path</span>
                  <input value={genOut} onChange={(e) => setGenOut(e.target.value)} disabled={isRunning} />
                </label>
                <label className="field">
                  <span>Distribution</span>
                  <select value={genDist} onChange={(e) => setGenDist(e.target.value)} disabled={isRunning}>
                    <option value="uniform">Uniform</option>
                    <option value="clustered">Clustered</option>
                    <option value="zipfian">Zipfian</option>
                  </select>
                </label>
                <label className="field">
                  <span>Classes (comma-sep)</span>
                  <input value={genClasses} onChange={(e) => setGenClasses(e.target.value)} disabled={isRunning} placeholder="A,B,C" />
                </label>
                <label className="field">
                  <span>Class Dist (optional)</span>
                  <input value={genClassDist} onChange={(e) => setGenClassDist(e.target.value)} disabled={isRunning} placeholder="0.6,0.3,0.1" />
                </label>
                <label className="field">
                  <span>Seed (optional)</span>
                  <input type="number" value={genSeed} onChange={(e) => setGenSeed(e.target.value)} disabled={isRunning} />
                </label>
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <button className="btn btnPrimary" onClick={startGenerate} disabled={isRunning}>
                  Generate Dataset
                </button>
              </div>
            </div>

            {/* Dataset List */}
            <div style={{ flex: 1 }}>
              <h4 style={{ marginBottom: 8 }}>Available Datasets</h4>
              <div style={{ maxHeight: 200, overflow: 'auto', background: 'var(--bg-subtle)', padding: 8, borderRadius: 4 }}>
                {datasets.length === 0 ? (
                  <div className="subtle">(no datasets found)</div>
                ) : (
                  <table style={{ width: '100%', fontSize: 13 }}>
                    <thead>
                      <tr><th>Name</th><th>Vectors</th><th>Dim</th></tr>
                    </thead>
                    <tbody>
                      {datasets.map((d) => (
                        <tr key={d.name}>
                          <td><code>{d.name}</code></td>
                          <td>{d.count?.toLocaleString() || '?'}</td>
                          <td>{d.dimensions || '?'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
              <button className="btn btnSmall" onClick={loadDatasets} style={{ marginTop: 8 }}>
                Refresh List
              </button>
            </div>
          </div>
        </section>

        {/* ============================================================ */}
        {/* PART 3: INGESTION BENCHMARK */}
        {/* ============================================================ */}
        <section className="card span2">
          <div className="cardHeader">
            <h3>Ingestion Benchmark</h3>
            <span className="subtle">Test data loading performance</span>
          </div>

          <div className="form">
            <label className="field">
              <span>Dataset</span>
              <select value={ingestDataset} onChange={(e) => setIngestDataset(e.target.value)} disabled={isRunning}>
                {datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Batch Size</span>
              <input type="number" value={ingestBatchSize} onChange={(e) => setIngestBatchSize(Number(e.target.value))} disabled={isRunning} />
            </label>
            <label className="field">
              <span>Index Type</span>
              <select value={ingestIndexType} onChange={(e) => setIngestIndexType(e.target.value)} disabled={isRunning}>
                <option value="ivfflat">IVFFlat</option>
                <option value="hnsw">HNSW</option>
              </select>
            </label>
            {ingestIndexType === 'ivfflat' ? (
              <label className="field">
                <span>Lists</span>
                <input type="number" value={ingestLists} onChange={(e) => setIngestLists(Number(e.target.value))} disabled={isRunning} />
              </label>
            ) : (
              <>
                <label className="field">
                  <span>HNSW M</span>
                  <input type="number" value={ingestHnswM} onChange={(e) => setIngestHnswM(Number(e.target.value))} disabled={isRunning} />
                </label>
                <label className="field">
                  <span>HNSW ef</span>
                  <input type="number" value={ingestHnswEf} onChange={(e) => setIngestHnswEf(Number(e.target.value))} disabled={isRunning} />
                </label>
              </>
            )}
            <label className="field" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="checkbox" checked={ingestFresh} onChange={(e) => setIngestFresh(e.target.checked)} disabled={isRunning} />
              <span>Fresh (clear previous results)</span>
            </label>
          </div>

          <div className="row" style={{ marginTop: 12 }}>
            <button className="btn btnPrimary" onClick={startIngestion} disabled={isRunning || !ingestDataset}>
              Run Ingestion Benchmark
            </button>
            <div className="progressWrap" aria-hidden="true">{progress}</div>
          </div>
        </section>

        <section className="card span2">
          <div className="cardHeader">
            <h3>Ingestion Plots</h3>
            <span className="subtle">Generate and view ingestion result charts</span>
          </div>

          <div style={{ marginBottom: 12 }}>
            <h4>Generate Plots</h4>
            <div className="row wrap">
              {['throughput_comparison', 'storage_comparison', 'memory_usage', 'time_breakdown', 'dimensionality_impact', 'resource_usage', 'index_storage_breakdown', 'index_build_time'].map((name) => (
                <button key={name} className="pill" onClick={() => generatePlot('ingestion', name)} disabled={isRunning}>
                  {name}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h4>Generated Plots</h4>
            <div className="plotScroll">
              {ingestionPlots.length === 0 ? (
                <div className="subtle">(no plots yet)</div>
              ) : (
                ingestionPlots.map((p) => (
                  <div key={p} className="plotCard">
                    <button
                      className="plotDelete"
                      onClick={async () => {
                        await fetch(`${API}/plots/ingestion/${p}`, { method: 'DELETE' })
                        loadPlots()
                      }}
                      title="Delete plot"
                    >
                      ×
                    </button>
                    <img src={`${API}/plots/ingestion/${p}`} alt={p} />
                    <div className="plotLabel">{p}</div>
                  </div>
                ))
              )}
            </div>
            <button className="btn btnSmall" onClick={loadPlots} style={{ marginTop: 8 }}>
              Refresh Plots
            </button>
          </div>
        </section>

        {/* ============================================================ */}
        {/* PART 5: QUERY BENCHMARK */}
        {/* ============================================================ */}
        <section className="card span2">
          <div className="cardHeader">
            <h3>Query Benchmark</h3>
            <span className="subtle">Test search performance and recall</span>
          </div>

          <div className="form">
            <label className="field">
              <span>Dataset</span>
              <select value={queryDataset} onChange={(e) => setQueryDataset(e.target.value)} disabled={isRunning}>
                {datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Mode</span>
              <select value={queryMode} onChange={(e) => setQueryMode(e.target.value)} disabled={isRunning}>
                <option value="both">Both (filter + nofilter)</option>
                <option value="nofilter">No Filter</option>
                <option value="filter">With Filter</option>
              </select>
            </label>
            <label className="field">
              <span>Index Type</span>
              <select value={queryIndexType} onChange={(e) => setQueryIndexType(e.target.value)} disabled={isRunning}>
                <option value="ivfflat">IVFFlat</option>
                <option value="hnsw">HNSW</option>
              </select>
            </label>
            <label className="field">
              <span>Metric</span>
              <select value={queryMetric} onChange={(e) => setQueryMetric(e.target.value)} disabled={isRunning}>
                <option value="euclidean">Euclidean</option>
                <option value="cosine">Cosine</option>
                <option value="inner_product">Inner Product</option>
              </select>
            </label>
            <label className="field" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="checkbox" checked={queryNoRestart} onChange={(e) => setQueryNoRestart(e.target.checked)} disabled={isRunning} />
              <span>No Restart (skip container restart)</span>
            </label>
            <label className="field" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="checkbox" checked={queryFresh} onChange={(e) => setQueryFresh(e.target.checked)} disabled={isRunning} />
              <span>Fresh (clear previous results)</span>
            </label>
          </div>

          <div className="row" style={{ marginTop: 12 }}>
            <button className="btn btnPrimary" onClick={startQuery} disabled={isRunning || !queryDataset}>
              Run Query Benchmark
            </button>
            <div className="progressWrap" aria-hidden="true">{progress}</div>
          </div>
        </section>

        <section className="card span2">
          <div className="cardHeader">
            <h3>Query Plots</h3>
            <span className="subtle">Generate and view query result charts</span>
          </div>

          <div style={{ marginBottom: 12 }}>
            <h4>Generate Plots</h4>
            <div className="row wrap">
              {['latency_comparison', 'recall_comparison', 'latency_vs_recall', 'topk_impact', 'filter_impact', 'scaling_analysis', 'p99_latency', 'throughput', 'combined_summary'].map((name) => (
                <button key={name} className="pill" onClick={() => generatePlot('queries', name)} disabled={isRunning}>
                  {name}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h4>Generated Plots</h4>
            <div className="plotScroll">
              {queryPlots.length === 0 ? (
                <div className="subtle">(no plots yet)</div>
              ) : (
                queryPlots.map((p) => (
                  <div key={p} className="plotCard">
                    <button
                      className="plotDelete"
                      onClick={async () => {
                        await fetch(`${API}/plots/queries/${p}`, { method: 'DELETE' })
                        loadPlots()
                      }}
                      title="Delete plot"
                    >
                      ×
                    </button>
                    <img src={`${API}/plots/queries/${p}`} alt={p} />
                    <div className="plotLabel">{p}</div>
                  </div>
                ))
              )}
            </div>
            <button className="btn btnSmall" onClick={loadPlots} style={{ marginTop: 8 }}>
              Refresh Plots
            </button>
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
