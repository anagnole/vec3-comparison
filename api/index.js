const express = require('express');
const cors = require('cors');
const path = require('path');
const { runCommand, getJob, listJobs } = require('./jobRunner');

const app = express();
app.use(cors());
app.use(express.json());

const fs = require('fs');
const PROJECT_ROOT = path.join(__dirname, '..');
let PY = process.env.PYTHON_CMD || 'python3';

const venvCandidates = [
  path.join(PROJECT_ROOT, '.venv', 'bin', 'python'),
  path.join(PROJECT_ROOT, '.venv', 'bin', 'python3'),
  path.join(PROJECT_ROOT, '.venv', 'bin', 'python3.11'),
];
for (const p of venvCandidates) {
  if (fs.existsSync(p)) {
    PY = p;
    break;
  }
}

const WEB_RESULTS_DIR = path.join(PROJECT_ROOT, 'results', 'web');
const WEB_INGESTION_RESULTS = path.join(WEB_RESULTS_DIR, 'ingestion', 'results.json');
const WEB_QUERY_RESULTS = path.join(WEB_RESULTS_DIR, 'queries', 'results.json');
const WEB_INGESTION_PLOTS = path.join(WEB_RESULTS_DIR, 'plots', 'ingestion');
const WEB_QUERY_PLOTS = path.join(WEB_RESULTS_DIR, 'plots', 'queries');

// Ensure web results directories exist
[WEB_RESULTS_DIR, 
 path.join(WEB_RESULTS_DIR, 'ingestion'),
 path.join(WEB_RESULTS_DIR, 'queries'),
 path.join(WEB_RESULTS_DIR, 'plots', 'ingestion'),
 path.join(WEB_RESULTS_DIR, 'plots', 'queries')
].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// ============================================================
// PART 1: DOCKER CONTROLS
// ============================================================

app.post('/docker/:action', (req, res) => {
  const action = req.params.action;
  if (action === 'up') {
    const id = runCommand('docker', ['compose', 'up', '-d'], { cwd: PROJECT_ROOT });
    res.json({ jobId: id });
  } else if (action === 'down') {
    const id = runCommand('docker', ['compose', 'down'], { cwd: PROJECT_ROOT });
    res.json({ jobId: id });
  } else if (action === 'restart') {
    const id = runCommand('docker', ['compose', 'restart'], { cwd: PROJECT_ROOT });
    res.json({ jobId: id });
  } else {
    res.status(400).json({ error: 'action must be up|down|restart' });
  }
});

app.get('/docker/status', async (req, res) => {
  const { exec } = require('child_process');
  exec('docker compose ps --format json', { cwd: PROJECT_ROOT }, (err, stdout, stderr) => {
    if (err) {
      return res.json({ chroma: null, pgvector: null, error: stderr });
    }
    try {
      // docker compose ps --format json outputs one JSON per line
      const containers = stdout.trim().split('\n')
        .filter(line => line.trim())
        .map(line => {
          try { return JSON.parse(line); } catch { return null; }
        })
        .filter(Boolean);
      
      // Extract status for each container
      let chromaStatus = null;
      let pgvectorStatus = null;
      
      for (const c of containers) {
        const name = (c.Name || c.name || '').toLowerCase();
        const state = (c.State || c.state || '').toLowerCase();
        if (name.includes('chroma')) {
          chromaStatus = state;
        } else if (name.includes('pgvector') || name.includes('postgres')) {
          pgvectorStatus = state;
        }
      }
      
      res.json({ chroma: chromaStatus, pgvector: pgvectorStatus, containers });
    } catch (e) {
      res.json({ chroma: null, pgvector: null, raw: stdout });
    }
  });
});

// ============================================================
// PART 2: DATA GENERATION & LISTING
// ============================================================

app.get('/datasets', (req, res) => {
  const dataDir = path.join(PROJECT_ROOT, 'data');
  if (!fs.existsSync(dataDir)) {
    return res.json({ datasets: [] });
  }
  
  const datasets = [];
  const entries = fs.readdirSync(dataDir, { withFileTypes: true });
  
  for (const entry of entries) {
    if (entry.isDirectory()) {
      const dataPath = path.join(dataDir, entry.name);
      const metaPath = path.join(dataPath, 'meta.json');
      
      let meta = null;
      if (fs.existsSync(metaPath)) {
        try {
          meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
        } catch {}
      }
      
      // Check for vectors.npy
      const vectorsPath = path.join(dataPath, 'vectors.npy');
      const hasVectors = fs.existsSync(vectorsPath);
      
      datasets.push({
        name: entry.name,
        path: `data/${entry.name}`,
        hasVectors,
        count: meta?.size || meta?.count || null,
        dimensions: meta?.dim || meta?.dimensions || null,
        meta,
      });
    }
  }
  
  // Sort by size if available
  datasets.sort((a, b) => {
    const sizeA = a.count || 0;
    const sizeB = b.count || 0;
    return sizeA - sizeB;
  });
  
  res.json({ datasets });
});

app.post('/generate', (req, res) => {
  const { 
    size = 10000, 
    dim = 128, 
    out = 'data/test',
    distribution = 'gaussian',
    classes = ['A', 'B', 'C'],
    classDist = [0.1, 0.3, 0.6],
    seed = 42
  } = req.body || {};
  
  const script = path.join(PROJECT_ROOT, 'vec3', 'generate_data.py');
  const args = [
    script,
    '--size', String(size),
    '--dim', String(dim),
    '--out', out,
    '--distribution', distribution,
    '--seed', String(seed),
  ];
  
  // Add classes if provided
  if (classes && classes.length > 0) {
    args.push('--classes', ...classes);
  }
  
  // Add class distribution if provided
  if (classDist && classDist.length > 0) {
    args.push('--class-dist', ...classDist.map(String));
  }
  
  const id = runCommand(PY, args, { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

// ============================================================
// PART 3: INGESTION BENCHMARKS
// ============================================================

app.post('/benchmark/ingest', (req, res) => {
  const {
    dataset,
    batchSize = 1000,
    indexType = 'hnsw',
    lists = 100,
    hnswM = 16,
    hnswEf = 64,
    fresh = false,
  } = req.body || {};
  
  if (!dataset) {
    return res.status(400).json({ error: 'dataset is required' });
  }
  
  const script = path.join(PROJECT_ROOT, 'benchmarks', 'ingestion', 'run_single_dataset.py');
  const args = [
    script,
    '--dataset', dataset,
    '--batch-size', String(batchSize),
    '--index-type', indexType,
    '--lists', String(lists),
    '--hnsw-m', String(hnswM),
    '--hnsw-ef', String(hnswEf),
    '--results-file', WEB_INGESTION_RESULTS,
  ];
  
  if (fresh) {
    args.push('--fresh');
  }
  
  const id = runCommand(PY, args, { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.get('/results/ingestion', (req, res) => {
  if (!fs.existsSync(WEB_INGESTION_RESULTS)) {
    return res.json({ runs: [] });
  }
  try {
    const data = JSON.parse(fs.readFileSync(WEB_INGESTION_RESULTS, 'utf8'));
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: 'Failed to parse results', detail: e.message });
  }
});

app.delete('/results/ingestion', (req, res) => {
  if (fs.existsSync(WEB_INGESTION_RESULTS)) {
    fs.unlinkSync(WEB_INGESTION_RESULTS);
  }
  res.json({ ok: true });
});

// ============================================================
// PART 4: INGESTION PLOTS
// ============================================================

const INGESTION_PLOTS = [
  'throughput_comparison',
  'storage_comparison',
  'memory_usage',
  'time_breakdown',
  'dimensionality_impact',
  'resource_usage',
  'index_storage_breakdown',
  'index_build_time',
];

app.get('/plots/ingestion', (req, res) => {
  const available = [];
  for (const name of INGESTION_PLOTS) {
    const file = path.join(WEB_INGESTION_PLOTS, `${name}.png`);
    available.push({
      name,
      exists: fs.existsSync(file),
      url: `/plots/ingestion/${name}.png`,
    });
  }
  res.json({ plots: available });
});

app.post('/plots/ingestion/:name', (req, res) => {
  const name = req.params.name;
  if (!INGESTION_PLOTS.includes(name)) {
    return res.status(400).json({ error: `Unknown plot: ${name}. Available: ${INGESTION_PLOTS.join(', ')}` });
  }
  
  // Run the plotting script with web paths
  const plotScript = `
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
import os
os.makedirs('${WEB_INGESTION_PLOTS}', exist_ok=True)

# Override paths
import benchmarks.plotting.ingestion_plots as ip
ip.RESULTS_FILE = '${WEB_INGESTION_RESULTS}'
ip.PLOTS_DIR = '${WEB_INGESTION_PLOTS}'

data = ip.load_results()
ip.plot_${name}(data)
print('Generated: ${name}.png')
`;
  
  const id = runCommand(PY, ['-c', plotScript], { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.post('/plots/ingestion/all', (req, res) => {
  const plotScript = `
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
import os
os.makedirs('${WEB_INGESTION_PLOTS}', exist_ok=True)

import benchmarks.plotting.ingestion_plots as ip
ip.RESULTS_FILE = '${WEB_INGESTION_RESULTS}'
ip.PLOTS_DIR = '${WEB_INGESTION_PLOTS}'

data = ip.load_results()
print(f"Loaded {len(data['runs'])} runs")

ip.plot_throughput_comparison(data)
ip.plot_storage_comparison(data)
ip.plot_memory_usage(data)
ip.plot_time_breakdown(data)
ip.plot_dimensionality_impact(data)
ip.plot_resource_usage(data)
ip.plot_index_storage_breakdown(data)
ip.plot_index_build_time(data)
print('All ingestion plots generated')
`;
  
  const id = runCommand(PY, ['-c', plotScript], { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.get('/plots/ingestion/:filename', (req, res) => {
  const filename = req.params.filename;
  const file = path.join(WEB_INGESTION_PLOTS, filename);
  if (!fs.existsSync(file)) {
    return res.status(404).json({ error: 'Plot not found' });
  }
  res.sendFile(file);
});

// ============================================================
// PART 5: QUERY BENCHMARKS
// ============================================================

app.post('/benchmark/query', (req, res) => {
  const {
    dataset,
    mode = 'nofilter', // 'nofilter', 'filter', 'both'
    indexType = 'hnsw',
    lists = 100,
    hnswM = 16,
    hnswEf = 64,
    metric = 'euclidean',
    noRestart = false,
    fresh = false,
  } = req.body || {};
  
  if (!dataset) {
    return res.status(400).json({ error: 'dataset is required' });
  }
  
  const script = path.join(PROJECT_ROOT, 'benchmarks', 'queries', 'run_single_dataset.py');
  const args = [
    script,
    dataset,
    '--index-type', indexType,
    '--lists', String(lists),
    '--hnsw-m', String(hnswM),
    '--hnsw-ef', String(hnswEf),
    '--metric', metric,
    '--results-file', WEB_QUERY_RESULTS,
  ];
  
  if (mode === 'filter') {
    args.push('--filter');
  } else if (mode === 'both') {
    args.push('--both');
  }
  
  if (noRestart) {
    args.push('--no-restart');
  }
  
  if (fresh) {
    args.push('--fresh');
  }
  
  const id = runCommand(PY, args, { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.get('/results/queries', (req, res) => {
  if (!fs.existsSync(WEB_QUERY_RESULTS)) {
    return res.json({ runs: [] });
  }
  try {
    const data = JSON.parse(fs.readFileSync(WEB_QUERY_RESULTS, 'utf8'));
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: 'Failed to parse results', detail: e.message });
  }
});

app.delete('/results/queries', (req, res) => {
  if (fs.existsSync(WEB_QUERY_RESULTS)) {
    fs.unlinkSync(WEB_QUERY_RESULTS);
  }
  res.json({ ok: true });
});

// ============================================================
// PART 5b: QUERY PLOTS
// ============================================================

const QUERY_PLOTS = [
  'latency_comparison',
  'recall_comparison',
  'latency_vs_recall',
  'topk_impact',
  'filter_impact',
  'scaling_analysis',
  'p99_latency',
  'throughput',
  'combined_summary',
];

app.get('/plots/queries', (req, res) => {
  const available = [];
  for (const name of QUERY_PLOTS) {
    const file = path.join(WEB_QUERY_PLOTS, `${name}.png`);
    available.push({
      name,
      exists: fs.existsSync(file),
      url: `/plots/queries/${name}.png`,
    });
  }
  res.json({ plots: available });
});

app.post('/plots/queries/:name', (req, res) => {
  const name = req.params.name;
  if (!QUERY_PLOTS.includes(name)) {
    return res.status(400).json({ error: `Unknown plot: ${name}. Available: ${QUERY_PLOTS.join(', ')}` });
  }
  
  const plotScript = `
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
import os
os.makedirs('${WEB_QUERY_PLOTS}', exist_ok=True)

import benchmarks.plotting.queries_plots as qp
qp.RESULTS_FILE = '${WEB_QUERY_RESULTS}'
qp.PLOTS_DIR = '${WEB_QUERY_PLOTS}'

data = qp.load_results()
qp.plot_${name}(data)
print('Generated: ${name}.png')
`;
  
  const id = runCommand(PY, ['-c', plotScript], { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.post('/plots/queries/all', (req, res) => {
  const plotScript = `
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
import os
os.makedirs('${WEB_QUERY_PLOTS}', exist_ok=True)

import benchmarks.plotting.queries_plots as qp
qp.RESULTS_FILE = '${WEB_QUERY_RESULTS}'
qp.PLOTS_DIR = '${WEB_QUERY_PLOTS}'

data = qp.load_results()
print(f"Loaded {len(data['runs'])} runs")

qp.plot_latency_comparison(data)
qp.plot_recall_comparison(data)
qp.plot_latency_vs_recall(data)
qp.plot_topk_impact(data)
qp.plot_filter_impact(data)
qp.plot_scaling_analysis(data)
qp.plot_p99_latency(data)
qp.plot_throughput_comparison(data)
qp.plot_combined_summary(data)
print('All query plots generated')
`;
  
  const id = runCommand(PY, ['-c', plotScript], { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.get('/plots/queries/:filename', (req, res) => {
  const filename = req.params.filename;
  const file = path.join(WEB_QUERY_PLOTS, filename);
  if (!fs.existsSync(file)) {
    return res.status(404).json({ error: 'Plot not found' });
  }
  res.sendFile(file);
});

// ============================================================
// JOB MANAGEMENT
// ============================================================

app.get('/jobs', (req, res) => {
  res.json(listJobs());
});

app.get('/jobs/:id', (req, res) => {
  const j = getJob(req.params.id);
  if (!j) return res.status(404).json({ error: 'job not found' });
  res.json(j);
});

// ============================================================
// LEGACY ENDPOINTS (kept for backwards compatibility)
// ============================================================

app.post('/ingest/:target', (req, res) => {
  const target = req.params.target;
  const { data_dir } = req.body || {};
  if (!data_dir) return res.status(400).json({ error: 'data_dir required' });

  let scriptName;
  if (target === 'chroma') scriptName = 'ingest_chroma.py';
  else if (target === 'pgvector') scriptName = 'ingest_pgvector.py';
  else return res.status(400).json({ error: 'unknown target' });

  const script = path.join(PROJECT_ROOT, 'vec3', scriptName);
  const id = runCommand(PY, [script, '--data-dir', data_dir], { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.get('/metrics', (req, res) => {
  const metricsPath = path.join(PROJECT_ROOT, 'results', 'raw', 'all_ingestion_results.json');
  if (!fs.existsSync(metricsPath)) return res.status(404).json({ error: 'metrics not found' });
  res.sendFile(metricsPath);
});

app.get('/plots', (req, res) => {
  const plotsDir = path.join(PROJECT_ROOT, 'results', 'plots');
  if (!fs.existsSync(plotsDir)) return res.json({ plots: [] });
  const files = fs.readdirSync(plotsDir).filter(f => f.endsWith('.png') || f.endsWith('.jpg'));
  res.json({ plots: files });
});

app.get('/plots/:name', (req, res) => {
  const name = req.params.name;
  const p = path.join(PROJECT_ROOT, 'results', 'plots', name);
  if (!fs.existsSync(p)) return res.status(404).json({ error: 'not found' });
  res.sendFile(p);
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`vec3 API running on http://localhost:${PORT}`);
  console.log(`  Web results: ${WEB_RESULTS_DIR}`);
});
