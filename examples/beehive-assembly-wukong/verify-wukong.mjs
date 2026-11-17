// Headless Chrome CDP smoke test for the BeeHive Wukong demo.
// Launches Chrome via the debug pipe (no puppeteer needed), loads the page,
// waits for the FBX to load (polls #wukong-status), captures console errors,
// reads the live phase label, and saves a screenshot into the temp dir.
//
// Usage:  node verify-wukong.mjs  <url>
//   e.g.  node verify-wukong.mjs http://localhost:8173/index.html
//
// Exit 0 = FBX loaded + phase advancing + zero fatal module errors.
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import os from 'node:os';
import { join } from 'node:path';

const CHROME = process.env.CHROME
  || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const URL = process.argv[2] || 'http://localhost:8173/index.html';
const SHOT = join(os.tmpdir(), 'kilo', 'wukong-shot.png');
const TIMEOUT_MS = 70000;

// launch headless chrome on a random debug port (global)
const port = 9300 + Math.floor(Math.random() * 500);
const userData = join(os.tmpdir(), 'kilo', `wukong-prof-${Date.now()}`);
const chrome = spawn(CHROME, [
  '--headless=new', '--no-sandbox', '--disable-gpu',
  '--window-size=1600,900', `--remote-debugging-port=${port}`,
  `--user-data-dir=${userData}`, 'about:blank'
], { stdio: 'ignore' });

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function json(url) {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 8000);
  try { const r = await fetch(url, { signal: ctl.signal }); return await r.json(); }
  finally { clearTimeout(t); }
}

let wsUrl = null;
const deadline = Date.now() + 15000;
while (Date.now() < deadline) {
  try {
    const list = await json(`http://127.0.0.1:${port}/json`);
    const page = list.find(t => t.type === 'page');
    if (page?.webSocketDebuggerUrl) { wsUrl = page.webSocketDebuggerUrl; break; }
  } catch (_) {}
  await sleep(250);
}
if (!wsUrl) { console.error('NO_WS: chrome did not open a page target'); chrome.kill(); process.exit(2); }

// minimal CDP client
let msgId = 0; const pending = new Map();
const ws = new WebSocket(wsUrl);
const call = (method, params = {}) => new Promise((res, rej) => {
  const id = ++msgId; pending.set(id, res);
  ws.send(JSON.stringify({ id, method, params }));
  setTimeout(() => { if (pending.delete(id)) rej(new Error('cdp timeout ' + method)); }, 15000);
});

ws.onopen = () => {};
const consoleErrors = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  else if (m.method === 'Runtime.exceptionThrown') {
    const ex = m.params?.exceptionDetails?.exception?.description || m.params?.exceptionDetails?.text || '?';
    consoleErrors.push(ex);
  } else if (m.method === 'Log.entryAdded' && m.params?.entry?.level === 'error') {
    consoleErrors.push('[log] ' + m.params.entry.text);
  }
};
await new Promise(r => { if (ws.readyState === 1) r(); else ws.onopen = r; });
await call('Page.enable'); await call('Runtime.enable'); await call('Log.enable');

const evaluate = async (expr) => {
  const r = await call('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  return r?.result?.value;
};

// navigate + wait for load
await call('Page.navigate', { url: URL });
await sleep(3000);

// poll for the wukong status marker (FBX decodes) and module boot (phase shows)
let status = '', phase = '', errorsSnap = '';
const start = Date.now();
while (Date.now() - start < TIMEOUT_MS) {
  status = await evaluate(`(document.getElementById('wukong-status')||{}).textContent || ''`);
  phase  = await evaluate(`(document.getElementById('phase')||{}).textContent || ''`);
  if (/ready|failed|unavailable/.test(status)) break;
  await sleep(2000);
}

// once FBX is ready, let the loop advance past the spawn so Wukong is on-screen
if (/ready/.test(status)) { await sleep(7000); }

// re-read live state (status now includes Wukong world position) + errors
status = await evaluate(`(document.getElementById('wukong-status')||{}).textContent || ''`);
phase  = await evaluate(`(document.getElementById('phase')||{}).textContent || ''`);
errorsSnap = JSON.stringify(consoleErrors);

// screenshot
try {
  const s = await call('Page.captureScreenshot', { format: 'png' });
  if (s?.data) writeFileSync(SHOT, Buffer.from(s.data, 'base64'));
} catch (e) { console.error('shot failed: ' + e.message); }

console.log('WUKONG_STATUS=' + (status || '<not-set>'));
console.log('PHASE=' + (phase || '<not-set>'));
console.log('CONSOLE_ERRORS=' + (errorsSnap || '[]'));
console.log('SCREENSHOT=' + SHOT);

const ok = /ready|wukong:\s+(enter|idle|orbit)/.test(status) && /phase/.test(phase) && !/not-set/.test(status+phase);
console.log(ok ? 'RESULT=PASS' : 'RESULT=FAIL');
chrome.kill();
process.exit(ok ? 0 : 1);