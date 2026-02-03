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

app.post('/generate', (req, res) => {
  const { size = 10000, dim = 128, out = 'data/test' } = req.body || {};
  const script = path.join(PROJECT_ROOT, 'vec3', 'generate_data.py');
  const id = runCommand(PY, [script, '--size', String(size), '--dim', String(dim), '--out', out], { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.post('/ingest/:target', (req, res) => {
  const target = req.params.target; // chroma or pgvector
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

app.post('/docker/:action', (req, res) => {
  const action = req.params.action;
  if (action !== 'up' && action !== 'down') return res.status(400).json({ error: 'action must be up|down' });
  const args = action === 'up' ? ['compose', 'up', '-d'] : ['compose', 'down'];
  const id = runCommand('docker', args, { cwd: PROJECT_ROOT });
  res.json({ jobId: id });
});

app.get('/jobs', (req, res) => {
  res.json(listJobs());
});

app.get('/jobs/:id', (req, res) => {
  const j = getJob(req.params.id);
  if (!j) return res.status(404).json({ error: 'job not found' });
  res.json(j);
});

// Serve generated plots and metrics
app.get('/metrics', (req, res) => {
  const metricsPath = path.join(PROJECT_ROOT, 'results', 'raw', 'all_ingestion_results_combined.json');
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
});
