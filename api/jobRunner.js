const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');

const jobs = {};

function runCommand(command, args, options = {}) {
  const id = uuidv4();
  const child = spawn(command, args, Object.assign({ shell: false }, options));

  jobs[id] = {
    id,
    command: [command].concat(args).join(' '),
    status: 'running',
    startedAt: new Date().toISOString(),
    finishedAt: null,
    exitCode: null,
    stdout: '',
    stderr: '',
    pid: child.pid,
  };

  child.stdout.on('data', (d) => {
    jobs[id].stdout += d.toString();
  });

  child.stderr.on('data', (d) => {
    jobs[id].stderr += d.toString();
  });

  child.on('close', (code) => {
    jobs[id].status = code === 0 ? 'completed' : 'failed';
    jobs[id].exitCode = code;
    jobs[id].finishedAt = new Date().toISOString();
  });

  child.on('error', (err) => {
    jobs[id].status = 'error';
    jobs[id].stderr += '\n' + err.toString();
    jobs[id].finishedAt = new Date().toISOString();
  });

  return id;
}

function getJob(id) {
  return jobs[id] || null;
}

function listJobs() {
  return Object.values(jobs);
}

module.exports = { runCommand, getJob, listJobs };
