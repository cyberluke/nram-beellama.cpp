// wukong-motion-server.mjs — local HTTP server that turns a natural-language prompt into
// an animated Wukong GLB via ComfyUI+HY-Motion, and PERSISTS each prompt+result as a
// "run" keyed by SHA256 (so you can A/B the best motion prompts).
//
// Endpoints:
//   GET  /health        -> { ok, comfy, runs }
//   GET  /runs          -> list of persisted run records (ordered newest first)
//   GET  /glb/<name>    -> serve a generated animated GLB
//   POST /generate      -> body JSON { prompt, seed?, duration?, run? } generates &
//                          persists a new run; returns the run record (blocking).
//   GET  /runs/<sha>    -> one run record by sha256
//   GET  /export/<name> -> download a generated GLB
//
// Data layout (in the SERVER's own dir):
//   runs.jsonl            — append-only run manifest (one JSON object per line)
//   glb/<run>-<sha>.glb   — the versioned animated output
//
// Start:  node tools/wukong-motion-server.mjs [port]   (default 8977)
// ComfyUI :8188 must be running (see start-comfy-kent.bat + PYTHONIOENCODING=utf-8).

import { createServer } from 'node:http';
import { randomUUID } from 'node:crypto';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync, copyFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dir = dirname(fileURLToPath(import.meta.url));
const RUNS = join(__dir, '..', 'wukong-motion');    // durable run records + GLBs
const GLB_DIR = join(RUNS, 'glb');
const MANIFEST = join(RUNS, 'runs.jsonl');
mkdirSync(GLB_DIR, { recursive: true });

const COMFY = process.env.COMFY_URL || 'http://127.0.0.1:8188';
// the lean web GLB that HYMotionExportGLB retargets the motion onto.
const LOCAL_CHAR = process.env.LOCAL_CHAR || join(__dir, '..', 'wukong-web.glb');
const COMFY_CHAR = process.env.COMFY_CHAR || 'F:/COMFY/ComfyUI/input/3d/char.glb';

const WF_TEMPLATE = JSON.parse(readFileSync(join(__dir, 'wukong-motion.api.json'), 'utf8'));

function sha(b64) { return createHash('sha256').update(b64).digest('hex'); }
function now() { return new Date().toISOString(); }

// ---- run manifest helpers ----
function loadRuns() {
  if (!existsSync(MANIFEST)) return [];
  return readFileSync(MANIFEST, 'utf8').split('\n').filter(l => l.trim()).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
}
function saveRun(rec) {
  writeFileSync(MANIFEST, JSON.stringify(rec) + '\n', { flag: 'a' });
}

async function comfyReachable() {
  try { const r = await fetch(`${COMFY}/system_stats`, { signal: AbortSignal.timeout(4000) }); return r.ok; } catch { return false; }
}

// submit the workflow to ComfyUI and wait, mirroring ComfyClient.submit/waitForOutput
async function runGeneration(prompt, { seed, duration, cfg = 5.0 }) {
  // patchable fields in the template: prompt, seed, duration, num_samples, cfg_scale, model, output_prefix
  const set = (k, v) => { const t = WF_TEMPLATE.patchable[k]; if (t) WF_TEMPLATE.nodes[t[0]].inputs[t[1]] = v; };
  set('prompt', prompt);
  set('seed', typeof seed === 'number' ? seed : Math.floor(Math.random() * 2 ** 31));
  const dur = typeof duration === 'number' ? duration : 6;
  set('duration', dur);
  set('num_samples', 1);
  set('cfg_scale', cfg);
  // Reporter hash + deterministic output prefix for this run
  const prefix = `wukong_${Date.now()}`;
  set('output_prefix', prefix);

  // ensure the lean web GLB is the retarget target (HYMotionExportGLB needs char.glb)
  try { copyFileSync(LOCAL_CHAR, COMFY_CHAR); } catch (e) { /* leave existing */ }

  const res = await fetch(`${COMFY}/prompt`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt: WF_TEMPLATE.nodes, client_id: 'wukong-server' }) });
  if (!res.ok) { const b = await res.text(); throw new Error(`submit failed ${res.status}: ${b.slice(0,300)}`); }
  const pid = (await res.json()).prompt_id;

  // poll via ComfyUI /history
  const deadline = Date.now() + (25 * 60 * 1000);
  let status = 'running';
  let outputs = [];
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 2000));
    try {
      const h = await (await fetch(`${COMFY}/history/${pid}`, { signal: AbortSignal.timeout(6000) })).json();
      const e = h[pid];
      if (e && e.status) {
        status = e.status.status_str || status;
        if (e.status.completed || status === 'success') { outputs = e.outputs || []; break; }
        if (status === 'error' || e.status.status_str === 'error') { throw new Error('generation error'); }
      }
    } catch (err) { if (String(err).includes('generation error')) throw err; /* transient */ }
  }
  if (status !== 'success') throw new Error(`timeout / not success (${status})`);
  return { pid, prefix, dur };
}

function fileExists(p) { try { return statSync(p).isFile(); } catch { return false; } }

// ---- job store: non-blocking /generate ====
const jobs = new Map();   // jobId -> { state, prompt, spec, run?, error?, createdAt, finishedAt? }

async function runJob(jobId) {
  const j = jobs.get(jobId);
  if (!j) return;
  j.state = 'running';
  try {
    const { pid, dur } = await runGeneration(j.prompt, j.spec);
    const outDir = join('F:/COMFY', 'ComfyUI', 'output', 'wukong_motion_glb');
    let glb = null;
    if (existsSync(outDir)) {
      const f = readdirSync(outDir).filter(n => n.endsWith('.glb')).map(n => ({ n, m: statSync(join(outDir, n)).mtimeMs })).sort((a, b) => b.m - a.m)[0];
      if (f) glb = join(outDir, f.n);
    }
    if (!glb) throw new Error('no GLB produced');
    const data = readFileSync(glb);
    const runId = sha(data);
    const fname = `wukong-${runId.slice(0, 12)}.glb`;
    copyFileSync(glb, join(GLB_DIR, fname));
    const rec = { sha: runId, prompt: j.prompt, params: j.spec || {}, file: `glb/${fname}`, pid, clip: 'Take 001', createdAt: now() };
    saveRun(rec);
    j.state = 'done'; j.run = rec; j.finishedAt = now();
  } catch (e) {
    console.error('[wukong-server] job', jobId, 'failed:', e instanceof Error ? e.stack || e.message : e);
    j.state = 'error'; j.error = String(e && e.message || e); j.finishedAt = now();
  }
}

const MIME = { '.glb': 'model/gltf-binary', '.json': 'application/json' };
function sendJson(res, code, obj) { const b = JSON.stringify(obj); res.writeHead(code, { 'content-type': 'application/json; charset=utf-8', 'content-length': Buffer.byteLength(b) }); res.end(b); }

// ---- HTTP server ----
// ---- CORS: allow the browser page (localhost:8173) to call this API cross-origin ----
function cors(res){ res.setHeader('Access-Control-Allow-Origin','*'); res.setHeader('Access-Control-Allow-Methods','GET,POST,OPTIONS'); res.setHeader('Access-Control-Allow-Headers','content-type'); }

const server = createServer(async (req, res) => {
  cors(res);
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
  const u = new URL(req.url, 'http://x');
  const path = decodeURIComponent(u.pathname);
  const method = req.method;
  try {
    if (method === 'GET' && path === '/health') {
      const comfy = await comfyReachable();
      return sendJson(res, 200, { ok: true, comfy, runs: loadRuns().length });
    }
    if (method === 'GET' && path === '/runs') {
      return sendJson(res, 200, { runs: loadRuns().reverse() });
    }
    if (method === 'GET' && path.startsWith('/runs/')) {
      const sha = path.slice('/runs/'.length);
      const run = loadRuns().find(r => r.sha === sha);
      return sendJson(res, run ? 200 : 404, run || { error: 'not found' });
    }
    if (method === 'GET' || method === 'HEAD') if (path.startsWith('/glb/')) {
      const name = path.slice('/glb/'.length);
      const file = join(GLB_DIR, name);
      if (!existsSync(file)) { if(method==='HEAD'){res.writeHead(404); res.end(); return;} return sendJson(res, 404, { error: 'missing' }); }
      const st = statSync(file);
      res.writeHead(200, { 'content-type': 'model/gltf-binary', 'content-length': st.size });
      if (method === 'HEAD') { res.end(); return; }
      res.end(readFileSync(file));
      return;
    }
    if (method === 'POST' && path === '/generate') {
      let body = '';
      for await (const c of req) body += c;
      let spec = {};
      try { spec = JSON.parse(body || '{}'); } catch {}
      const prompt = spec.prompt;
      if (!prompt || !String(prompt).trim()) return sendJson(res, 400, { error: 'prompt required' });
      // Non-blocking: return a jobId immediately; the run record lands in /runs when done.
      const jobId = randomUUID();
      jobs.set(jobId, { state: 'queued', prompt: String(prompt).trim(), spec, createdAt: now() });
      runJob(jobId).catch(e => { console.error('[wukong-server] job', jobId, 'failed', e); const j = jobs.get(jobId); if(j){ j.state='error'; j.error=String(e&&e.message||e); } });
      return sendJson(res, 202, { jobId, message: 'queued' });
    }
    if (method === 'GET' && path.startsWith('/jobs/')) {
      const jobId = path.slice('/jobs/'.length);
      const j = jobs.get(jobId);
      if (!j) return sendJson(res, 404, { error: 'no such job' });
      return sendJson(res, 200, j);
    }
    return sendJson(res, 404, { error: 'not found' });
  } catch (e) { sendJson(res, 500, { error: String(e && e.message || e) }); }
});

const PORT = process.env.PORT || Number(process.argv[2]) || 8977;
server.listen(PORT, () => console.log(`[wukong-server] listening on http://127.0.0.1:${PORT} · runs: ${RUNS}`));