"""Live experiment dashboard — polls all Vast.ai instances every 60 seconds.

Usage:
  python dashboard.py

Opens in browser at http://localhost:8050
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

INSTANCES = {
    "A": {"id": 33132302, "host": "ssh3.vast.ai", "port": 12302, "gpu": "RTX 4090", "task": "Full baselines + ViT-S SVHN n=5000", "log": "log_A.txt"},
    "B": {"id": 33132101, "host": "ssh8.vast.ai", "port": 12100, "gpu": "RTX 4090", "task": "ViT-B CIFAR-100 + DTD + SVHN n=500", "log": "log_B.txt"},
    "C": {"id": 33132103, "host": "ssh8.vast.ai", "port": 12102, "gpu": "A100 80GB", "task": "ViT-L all 3 pairs n=200", "log": "log_C.txt"},
    "D": {"id": 33132104, "host": "ssh8.vast.ai", "port": 12104, "gpu": "RTX 4090", "task": "MetaCLIP all 3 pairs n=200", "log": "log_D.txt"},
}

SSH_KEY = Path.home() / ".ssh" / "id_ed25519"


def poll_instance(label: str, info: dict) -> dict:
    """SSH into instance, get last 5 log lines + process status."""
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                "-i", str(SSH_KEY), "-p", str(info["port"]),
                f"root@{info['host']}",
                f"tail -8 /workspace/{info.get('log', 'log_' + label + '.txt')} 2>/dev/null; echo '|||'; ps aux | grep python | grep -v grep | wc -l; echo '|||'; ls /workspace/results_extended/*.json 2>/dev/null | wc -l"
            ],
            capture_output=True, timeout=15, encoding="utf-8", errors="replace"
        )
        stdout = result.stdout or ""
        parts = stdout.split("|||")
        log = parts[0].strip().split("\n") if len(parts) > 0 and parts[0].strip() else []
        # Filter out the vast.ai welcome message
        log = [l for l in log if "vast.ai" not in l and "Have fun" not in l and l.strip()]
        procs = int(parts[1].strip()) if len(parts) > 1 else -1
        files = int(parts[2].strip()) if len(parts) > 2 else 0

        if procs == 0:
            status = "DONE"
        elif procs > 0:
            status = "RUNNING"
        else:
            status = "UNKNOWN"

        return {
            "label": label,
            "status": status,
            "gpu": info["gpu"],
            "task": info["task"],
            "log": log[-6:],
            "result_files": files,
            "process_count": procs,
        }
    except Exception as e:
        return {
            "label": label,
            "status": "UNREACHABLE",
            "gpu": info["gpu"],
            "task": info["task"],
            "log": [str(e)],
            "result_files": 0,
            "process_count": -1,
        }


_instance_start_times: dict[str, float] = {}


@app.get("/api/status")
def get_status():
    results = {}
    now = time.time()
    for label, info in INSTANCES.items():
        inst = poll_instance(label, info)

        # Track first-seen running time
        if inst["status"] == "RUNNING" and label not in _instance_start_times:
            _instance_start_times[label] = now
        if inst["status"] == "DONE" and label not in _instance_start_times:
            _instance_start_times[label] = now

        # Compute elapsed
        start = _instance_start_times.get(label)
        inst["elapsed_s"] = int(now - start) if start else 0

        # Parse ETA from last log line containing "remaining"
        eta_text = ""
        for line in reversed(inst.get("log", [])):
            if "remaining" in line:
                import re
                m = re.search(r"~(\d+)min remaining", line)
                if m:
                    eta_text = f"~{m.group(1)} min"
                    break
                m = re.search(r"~(\d+)s remaining", line)  # noqa: E741
                if m:
                    eta_text = f"~{int(m.group(1))//60} min"
                    break
        inst["eta"] = eta_text

        results[label] = inst
    return {"instances": results, "timestamp": datetime.now().isoformat()}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html>
<head>
<title>Dokime Lab — Experiment Control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Inter', -apple-system, sans-serif; background: #09090b; color: #fafafa; min-height: 100vh; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }

  /* Header */
  .header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 28px; }
  .header-left h1 { font-size: 24px; font-weight: 700; background: linear-gradient(135deg, #7c3aed, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header-left .tagline { color: #71717a; font-size: 13px; margin-top: 4px; }
  .header-right { text-align: right; }
  .header-right .live { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: #a1a1aa; }
  .header-right .live .dot { width: 6px; height: 6px; background: #22c55e; border-radius: 50%; animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .header-right .timestamp { font-size: 11px; color: #52525b; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

  /* Stats bar */
  .stats { display: flex; gap: 12px; margin-bottom: 24px; }
  .stat-card { flex: 1; background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 16px; }
  .stat-card .label { font-size: 11px; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
  .stat-card .value { font-size: 28px; font-weight: 700; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
  .stat-card .value.green { color: #22c55e; }
  .stat-card .value.purple { color: #a78bfa; }
  .stat-card .value.amber { color: #f59e0b; }
  .stat-card .sub { font-size: 11px; color: #52525b; margin-top: 2px; }

  /* Grid */
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }

  /* Instance card */
  .card { background: #18181b; border: 1px solid #27272a; border-radius: 12px; overflow: hidden; transition: border-color 0.2s; }
  .card:hover { border-color: #3f3f46; }
  .card.done { border-color: #166534; }
  .card-top { padding: 16px 16px 12px; display: flex; justify-content: space-between; align-items: center; }
  .card-top h2 { font-size: 15px; font-weight: 600; }
  .card-top h2 .gpu-tag { font-size: 11px; font-weight: 500; color: #a78bfa; background: #7c3aed1a; padding: 2px 8px; border-radius: 6px; margin-left: 8px; }
  .badge { padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .badge.running { background: #1d4ed81a; color: #60a5fa; border: 1px solid #1d4ed833; }
  .badge.done { background: #16a34a1a; color: #4ade80; border: 1px solid #16a34a33; }
  .badge.unreachable { background: #dc26261a; color: #f87171; border: 1px solid #dc262633; }
  .card-meta { padding: 0 16px 12px; font-size: 12px; color: #a1a1aa; }
  .card-meta .task { color: #d4d4d8; font-weight: 500; }
  .card-meta .files { margin-top: 4px; }
  .card-meta .files span { color: #a78bfa; font-weight: 600; }

  /* Log terminal */
  .log-wrap { margin: 0 12px 12px; background: #09090b; border: 1px solid #1a1a1e; border-radius: 8px; overflow: hidden; }
  .log-header { padding: 6px 10px; background: #111113; border-bottom: 1px solid #1a1a1e; display: flex; align-items: center; gap: 6px; }
  .log-header .circles { display: flex; gap: 4px; }
  .log-header .circles span { width: 8px; height: 8px; border-radius: 50%; }
  .log-header .circles .r { background: #ef4444; }
  .log-header .circles .y { background: #f59e0b; }
  .log-header .circles .g { background: #22c55e; }
  .log-header .title { font-size: 10px; color: #52525b; font-family: 'JetBrains Mono', monospace; margin-left: 6px; }
  .log { padding: 10px 12px; font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.6; max-height: 140px; overflow-y: auto; }
  .log .line { color: #71717a; white-space: pre-wrap; word-break: break-all; }
  .log .line.recent { color: #d4d4d8; }
  .log .highlight { color: #f59e0b; font-weight: 600; }
  .log .score { color: #a78bfa; }

  /* Footer */
  .footer { margin-top: 24px; text-align: center; font-size: 11px; color: #3f3f46; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="header-left">
      <h1>Dokime Lab</h1>
      <div class="tagline">JEPA-SCORE Experiment Control &bull; 4 GPU instances</div>
    </div>
    <div class="header-right">
      <div class="live"><span class="dot"></span> Live</div>
      <div class="timestamp" id="last-update">connecting...</div>
      <div class="timestamp" id="next-poll">next poll in 30s</div>
    </div>
  </div>

  <div class="stats">
    <div class="stat-card"><div class="label">Running</div><div class="value purple" id="s-running">-</div></div>
    <div class="stat-card"><div class="label">Complete</div><div class="value green" id="s-done">-</div></div>
    <div class="stat-card"><div class="label">Elapsed</div><div class="value" id="s-elapsed">-</div><div class="sub" id="s-elapsed-sub"></div></div>
    <div class="stat-card"><div class="label">Est. Cost</div><div class="value amber" id="s-cost">-</div><div class="sub">of $12 budget</div></div>
  </div>

  <div class="grid" id="cards"></div>

  <div class="footer">Dokime &bull; Independent AI Research Lab &bull; Andrew Morgan</div>
</div>

<script>
const COSTS = {A: 0.302, B: 0.268, C: 1.015, D: 0.322};
const START = Date.now();

function fmt(n) { return n < 10 ? '0'+n : n; }

async function refresh() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const container = document.getElementById('cards');
    container.innerHTML = '';

    let running = 0, done = 0, totalRate = 0;

    for (const [label, inst] of Object.entries(data.instances)) {
      const st = inst.status.toLowerCase();
      if (st === 'running') running++;
      if (st === 'done') done++;
      if (st !== 'done') totalRate += COSTS[label] || 0.30;

      const logHtml = inst.log.map((line, i) => {
        const cls = i >= inst.log.length - 2 ? 'recent' : '';
        const hl = line.replace(/(AUROC=[0-9.]+)/g, '<span class="highlight">$1</span>')
                       .replace(/(score=[0-9.-]+)/g, '<span class="score">$1</span>')
                       .replace(/(DONE|Saved|COMPLETE)/gi, '<span class="highlight">$1</span>');
        return '<div class="line '+cls+'">'+hl+'</div>';
      }).join('') || '<div class="line">Waiting for logs...</div>';

      container.innerHTML += `
        <div class="card ${st === 'done' ? 'done' : ''}">
          <div class="card-top">
            <h2>Instance ${label}<span class="gpu-tag">${inst.gpu}</span></h2>
            <span class="badge ${st}">${inst.status}</span>
          </div>
          <div class="card-meta">
            <div class="task">${inst.task}</div>
            <div style="display:flex;gap:16px;margin-top:4px;">
              <div class="files">Results: <span>${inst.result_files}</span></div>
              <div class="files">Elapsed: <span>${Math.floor(inst.elapsed_s/60)}m ${inst.elapsed_s%60}s</span></div>
              <div class="files">ETA: <span>${inst.eta || (inst.status==='DONE'?'done':'...')}</span></div>
            </div>
          </div>
          <div class="log-wrap">
            <div class="log-header">
              <div class="circles"><span class="r"></span><span class="y"></span><span class="g"></span></div>
              <span class="title">log_${label}.txt</span>
            </div>
            <div class="log">${logHtml}</div>
          </div>
        </div>`;
    }

    document.getElementById('s-running').textContent = running;
    document.getElementById('s-done').textContent = done;

    const elapsed = (Date.now() - START) / 1000;
    const h = Math.floor(elapsed/3600), m = Math.floor((elapsed%3600)/60), s = Math.floor(elapsed%60);
    document.getElementById('s-elapsed').textContent = fmt(h)+':'+fmt(m)+':'+fmt(s);

    const estCost = (totalRate * elapsed / 3600).toFixed(2);
    document.getElementById('s-cost').textContent = '$' + estCost;
    document.getElementById('last-update').textContent = new Date(data.timestamp).toLocaleTimeString();

    if (done === 4) {
      document.querySelector('.live .dot').style.background = '#22c55e';
      document.querySelector('.tagline').textContent = 'ALL EXPERIMENTS COMPLETE';
    }
  } catch(e) { console.error(e); }
}

let lastPoll = Date.now();
const POLL_INTERVAL = 30000;

async function doPoll() {
  await refresh();
  lastPoll = Date.now();
}

doPoll();
setInterval(doPoll, POLL_INTERVAL);
setInterval(() => {
  const elapsed = (Date.now() - START) / 1000;
  const h = Math.floor(elapsed/3600), m = Math.floor((elapsed%3600)/60), s = Math.floor(elapsed%60);
  document.getElementById('s-elapsed').textContent = fmt(h)+':'+fmt(m)+':'+fmt(s);

  const nextIn = Math.max(0, Math.ceil((POLL_INTERVAL - (Date.now() - lastPoll)) / 1000));
  document.getElementById('next-poll').textContent = 'next poll in ' + nextIn + 's';
}, 1000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    print("Dashboard: http://localhost:8051")
    uvicorn.run(app, host="127.0.0.1", port=8051, log_level="warning")
