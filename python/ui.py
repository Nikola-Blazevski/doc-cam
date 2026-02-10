#!/usr/bin/env python3
"""
Face Snapshot Browser + Stats Dashboard.

Usage:
    python ui.py                    # default port 8080
    python ui.py --port 9000
    python ui.py --dir /path/to/snapshots --stats /path/to/face_stats.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import urllib.parse

DEFAULT_DIR = "face_snapshots"
DEFAULT_STATS = "face_stats.json"
DEFAULT_PORT = 8080


def get_snapshot_data(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return {"people": [], "base_dir": str(base)}
    people = []
    for uid_dir in sorted(base.iterdir()):
        if not uid_dir.is_dir():
            continue
        uid = uid_dir.name
        photos = []
        for img_file in sorted(uid_dir.iterdir(), reverse=True):
            if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            stem = img_file.stem
            try:
                dt = datetime.strptime(stem[:15], "%Y%m%d_%H%M%S")
                ms = stem[16:] if len(stem) > 15 else "000"
                ts_display = dt.strftime("%b %d, %Y  %H:%M:%S") + f".{ms}"
            except (ValueError, IndexError):
                ts_display = stem
            photos.append({
                "filename": img_file.name,
                "path": f"/snapshots/{uid}/{img_file.name}",
                "timestamp": ts_display,
                "raw_name": stem,
            })
        if photos:
            people.append({
                "uid": uid,
                "photo_count": len(photos),
                "photos": photos,
                "latest": photos[0]["path"],
            })
    people.sort(key=lambda p: p["photo_count"], reverse=True)
    return {"people": people, "base_dir": str(base)}


def get_stats_data(stats_path):
    if not os.path.exists(stats_path):
        return {}
    try:
        with open(stats_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Tracker — Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary: #0a0a0b;
    --bg-card: #111113;
    --bg-card-hover: #18181b;
    --bg-surface: #1a1a1e;
    --bg-overlay: rgba(10,10,11,0.92);
    --border: #27272a;
    --border-accent: #d97706;
    --text-primary: #e4e4e7;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent: #f59e0b;
    --accent-dim: #92400e;
    --accent-glow: rgba(245,158,11,0.12);
    --radius: 6px;
    --font-mono: 'JetBrains Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg-primary); color:var(--text-primary); font-family:var(--font-body); min-height:100vh; }
  body::before { content:''; position:fixed; inset:0; background-image:linear-gradient(var(--border) 1px,transparent 1px),linear-gradient(90deg,var(--border) 1px,transparent 1px); background-size:60px 60px; opacity:0.06; pointer-events:none; z-index:0; }

  .header { position:sticky; top:0; z-index:100; background:var(--bg-overlay); backdrop-filter:blur(16px); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .header-left { display:flex; align-items:center; gap:12px; }
  .logo-icon { width:30px; height:30px; background:linear-gradient(135deg,var(--accent),var(--accent-dim)); border-radius:7px; display:flex; align-items:center; justify-content:center; font-size:14px; color:#000; font-weight:700; }
  .header-title { font-family:var(--font-mono); font-size:14px; font-weight:600; } .header-title span { color:var(--accent); }
  .header-stats { font-family:var(--font-mono); font-size:11px; color:var(--text-muted); display:flex; gap:14px; }
  .stat-val { color:var(--accent); font-weight:600; }
  .header-actions { display:flex; gap:8px; }

  .tab-btn { background:transparent; border:1px solid var(--border); color:var(--text-secondary); padding:5px 14px; border-radius:var(--radius); font-family:var(--font-mono); font-size:11px; cursor:pointer; transition:all 0.15s; }
  .tab-btn:hover { border-color:var(--accent); color:var(--accent); background:var(--accent-glow); }
  .tab-btn.active { border-color:var(--accent); color:var(--accent); background:var(--accent-glow); }
  .refresh-btn { background:transparent; border:1px solid var(--border); color:var(--text-secondary); padding:5px 12px; border-radius:var(--radius); font-family:var(--font-mono); font-size:11px; cursor:pointer; transition:all 0.15s; }
  .refresh-btn:hover { border-color:var(--accent); color:var(--accent); }

  .container { position:relative; z-index:1; display:flex; height:calc(100vh - 57px); }

  /* Sidebar */
  .sidebar { width:260px; min-width:260px; border-right:1px solid var(--border); overflow-y:auto; background:var(--bg-card); }
  .sidebar-title { font-family:var(--font-mono); font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted); padding:14px 14px 6px; }
  .person-item { display:flex; align-items:center; gap:10px; padding:8px 14px; cursor:pointer; border-left:3px solid transparent; transition:all 0.12s; }
  .person-item:hover { background:var(--bg-card-hover); }
  .person-item.active { background:var(--bg-surface); border-left-color:var(--accent); }
  .person-avatar { width:36px; height:36px; border-radius:50%; object-fit:cover; border:2px solid var(--border); flex-shrink:0; background:var(--bg-surface); }
  .person-item.active .person-avatar { border-color:var(--accent); }
  .person-info { flex:1; min-width:0; }
  .person-uid { font-family:var(--font-mono); font-size:12px; font-weight:600; }
  .person-item.active .person-uid { color:var(--accent); }
  .person-count { font-size:10px; color:var(--text-muted); margin-top:1px; }

  /* Main */
  .main { flex:1; overflow-y:auto; padding:20px; }
  .empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:var(--text-muted); gap:10px; }
  .empty-state-icon { font-size:44px; opacity:0.25; }
  .empty-state-text { font-family:var(--font-mono); font-size:12px; }

  .person-header { display:flex; align-items:center; gap:14px; margin-bottom:20px; padding-bottom:14px; border-bottom:1px solid var(--border); }
  .person-header-avatar { width:48px; height:48px; border-radius:50%; object-fit:cover; border:2px solid var(--accent); background:var(--bg-surface); }
  .person-header-info h2 { font-family:var(--font-mono); font-size:18px; font-weight:600; color:var(--accent); }
  .person-header-info p { font-size:12px; color:var(--text-muted); margin-top:2px; }

  /* Photo grid */
  .photo-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:10px; }
  .photo-card { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; cursor:pointer; transition:all 0.15s; animation:fadeIn 0.2s ease both; }
  .photo-card:hover { border-color:var(--border-accent); transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,0.3); }
  .photo-card img { width:100%; aspect-ratio:1; object-fit:cover; display:block; background:var(--bg-surface); }
  .photo-card-info { padding:6px 8px; border-top:1px solid var(--border); }
  .photo-card-ts { font-family:var(--font-mono); font-size:9px; color:var(--text-muted); }

  /* Lightbox */
  .lightbox { display:none; position:fixed; inset:0; z-index:200; background:var(--bg-overlay); backdrop-filter:blur(20px); align-items:center; justify-content:center; flex-direction:column; gap:10px; cursor:pointer; }
  .lightbox.open { display:flex; }
  .lightbox img { max-width:80vw; max-height:75vh; border-radius:var(--radius); border:1px solid var(--border); box-shadow:0 16px 48px rgba(0,0,0,0.5); }
  .lightbox-meta { font-family:var(--font-mono); font-size:11px; color:var(--text-secondary); } .lightbox-meta span { color:var(--accent); font-weight:600; }
  .lightbox-close { position:absolute; top:16px; right:20px; font-size:20px; color:var(--text-muted); cursor:pointer; width:32px; height:32px; display:flex; align-items:center; justify-content:center; border-radius:50%; border:1px solid var(--border); transition:all 0.15s; }
  .lightbox-close:hover { color:var(--accent); border-color:var(--accent); }

  /* ---- Stats Section ---- */
  .stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }
  @media(max-width:900px) { .stats-grid { grid-template-columns:1fr; } }

  .stat-card { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:16px; }
  .stat-card-title { font-family:var(--font-mono); font-size:10px; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); margin-bottom:12px; }
  .stat-card-value { font-family:var(--font-mono); font-size:28px; font-weight:700; color:var(--accent); }
  .stat-card-sub { font-size:11px; color:var(--text-muted); margin-top:4px; }

  /* Bar chart */
  .hour-chart { display:flex; align-items:flex-end; gap:2px; height:120px; padding-top:8px; }
  .hour-bar-wrap { flex:1; display:flex; flex-direction:column; align-items:center; gap:2px; height:100%; justify-content:flex-end; }
  .hour-bar { width:100%; background:var(--accent); border-radius:2px 2px 0 0; min-height:1px; transition:height 0.3s; opacity:0.7; }
  .hour-bar.peak { opacity:1; }
  .hour-label { font-family:var(--font-mono); font-size:8px; color:var(--text-muted); }

  /* Heatmap (position) */
  .heatmap-container { position:relative; width:100%; aspect-ratio:16/9; background:var(--bg-surface); border-radius:var(--radius); overflow:hidden; border:1px solid var(--border); }
  .heatmap-dot { position:absolute; width:8px; height:8px; border-radius:50%; background:var(--accent); opacity:0.15; transform:translate(-50%,-50%); }
  .heatmap-center { position:absolute; width:16px; height:16px; border-radius:50%; border:2px solid var(--accent); background:rgba(245,158,11,0.3); transform:translate(-50%,-50%); }
  .heatmap-label { position:absolute; font-family:var(--font-mono); font-size:9px; color:var(--accent); transform:translate(-50%,12px); }
  .heatmap-axis { position:absolute; font-family:var(--font-mono); font-size:8px; color:var(--text-muted); }

  /* Day chart */
  .day-chart { display:flex; gap:6px; }
  .day-bar-wrap { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; }
  .day-bar { width:100%; height:40px; background:var(--bg-surface); border-radius:3px; position:relative; overflow:hidden; }
  .day-bar-fill { position:absolute; bottom:0; width:100%; background:var(--accent); border-radius:3px; opacity:0.6; transition:height 0.3s; }
  .day-label { font-family:var(--font-mono); font-size:9px; color:var(--text-muted); }
  .day-val { font-family:var(--font-mono); font-size:9px; color:var(--text-secondary); }

  ::-webkit-scrollbar { width:5px; } ::-webkit-scrollbar-track { background:transparent; } ::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo-icon">⬡</div>
    <div class="header-title">face<span>tracker</span></div>
  </div>
  <div class="header-stats">
    <span>identities: <span class="stat-val" id="stat-people">0</span></span>
    <span>captures: <span class="stat-val" id="stat-photos">0</span></span>
    <span>sightings: <span class="stat-val" id="stat-sightings">0</span></span>
  </div>
  <div class="header-actions">
    <button class="tab-btn active" id="tab-photos" onclick="switchTab('photos')">◫ photos</button>
    <button class="tab-btn" id="tab-stats" onclick="switchTab('stats')">◈ stats</button>
    <button class="refresh-btn" onclick="loadAll()">↻</button>
  </div>
</div>

<div class="container">
  <div class="sidebar">
    <div class="sidebar-title">identified persons</div>
    <div id="person-list"></div>
  </div>
  <div class="main" id="main-content">
    <div class="empty-state">
      <div class="empty-state-icon">◎</div>
      <div class="empty-state-text">select a person to view data</div>
    </div>
  </div>
</div>

<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <div class="lightbox-close" onclick="closeLightbox()">✕</div>
  <img id="lightbox-img" src="" alt="">
  <div class="lightbox-meta" id="lightbox-meta"></div>
</div>

<script>
let SNAP_DATA = { people: [] };
let STATS_DATA = {};
let selectedUid = null;
let currentTab = 'photos';

async function loadAll() {
  try {
    const [snapResp, statsResp] = await Promise.all([
      fetch('/api/snapshots'), fetch('/api/stats')
    ]);
    SNAP_DATA = await snapResp.json();
    STATS_DATA = await statsResp.json();
    renderSidebar();
    updateGlobalStats();
    if (selectedUid) refreshMain();
  } catch(e) { console.error('Load error:', e); }
}

function updateGlobalStats() {
  document.getElementById('stat-people').textContent = SNAP_DATA.people.length;
  const tp = SNAP_DATA.people.reduce((s,p) => s + p.photo_count, 0);
  document.getElementById('stat-photos').textContent = tp;
  let ts = 0;
  for (const uid in STATS_DATA) ts += (STATS_DATA[uid].sightings||[]).length;
  document.getElementById('stat-sightings').textContent = ts;
}

function renderSidebar() {
  const list = document.getElementById('person-list');
  // Merge snap people with stats-only people
  const allUids = new Set();
  SNAP_DATA.people.forEach(p => allUids.add(p.uid));
  for (const uid in STATS_DATA) allUids.add(uid);

  const items = [];
  allUids.forEach(uid => {
    const snap = SNAP_DATA.people.find(p => p.uid === uid);
    const stats = STATS_DATA[uid];
    const sightings = stats ? (stats.sightings||[]).length : 0;
    const photos = snap ? snap.photo_count : 0;
    items.push({ uid, photos, sightings, latest: snap ? snap.latest : null });
  });
  items.sort((a,b) => (b.sightings + b.photos) - (a.sightings + a.photos));

  list.innerHTML = '';
  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'person-item' + (item.uid === selectedUid ? ' active' : '');
    div.onclick = () => selectPerson(item.uid);
    const avatarHtml = item.latest
      ? `<img class="person-avatar" src="${item.latest}" onerror="this.style.display='none'">`
      : `<div class="person-avatar" style="display:flex;align-items:center;justify-content:center;font-size:14px;color:var(--text-muted)">?</div>`;
    div.innerHTML = `${avatarHtml}<div class="person-info"><div class="person-uid">${item.uid}</div><div class="person-count">${item.photos} photos · ${item.sightings} sightings</div></div>`;
    list.appendChild(div);
  });
}

function selectPerson(uid) { selectedUid = uid; renderSidebar(); refreshMain(); }
function switchTab(tab) { currentTab = tab; document.getElementById('tab-photos').classList.toggle('active', tab==='photos'); document.getElementById('tab-stats').classList.toggle('active', tab==='stats'); refreshMain(); }
function refreshMain() { if (!selectedUid) return; if (currentTab === 'photos') renderPhotos(); else renderStats(); }

function renderPhotos() {
  const person = SNAP_DATA.people.find(p => p.uid === selectedUid);
  const main = document.getElementById('main-content');
  if (!person) { main.innerHTML = '<div class="empty-state"><div class="empty-state-text">no photos for this person yet</div></div>'; return; }
  let h = `<div class="person-header"><img class="person-header-avatar" src="${person.latest}" onerror="this.style.display='none'"><div class="person-header-info"><h2>${person.uid}</h2><p>${person.photo_count} captures</p></div></div><div class="photo-grid">`;
  person.photos.forEach((photo,i) => {
    h += `<div class="photo-card" style="animation-delay:${i*25}ms" onclick="openLightbox('${photo.path}','${person.uid}','${photo.timestamp.replace(/'/g,"\\'")}')"><img src="${photo.path}" loading="lazy"><div class="photo-card-info"><div class="photo-card-ts">${photo.timestamp}</div></div></div>`;
  });
  main.innerHTML = h + '</div>';
}

function renderStats() {
  const main = document.getElementById('main-content');
  const stats = STATS_DATA[selectedUid];
  const snap = SNAP_DATA.people.find(p => p.uid === selectedUid);
  const latestImg = snap ? snap.latest : '';

  if (!stats || !stats.sightings || stats.sightings.length === 0) {
    main.innerHTML = '<div class="empty-state"><div class="empty-state-text">no stats recorded for this person yet</div></div>';
    return;
  }

  const sightings = stats.sightings;
  const positions = stats.positions || [];

  // Hourly distribution
  const hours = new Array(24).fill(0);
  sightings.forEach(s => hours[s.hour]++);
  const maxHour = Math.max(...hours, 1);
  const peakHour = hours.indexOf(maxHour);

  // Day of week distribution
  const days = new Array(7).fill(0);
  sightings.forEach(s => { if (s.weekday !== undefined) days[s.weekday]++; });
  const maxDay = Math.max(...days, 1);
  const dayNames = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const peakDay = dayNames[days.indexOf(Math.max(...days))];

  // Unique days seen
  const uniqueDays = new Set(sightings.map(s => s.day)).size;

  // Average position
  let avgX = 0.5, avgY = 0.5;
  if (positions.length > 0) {
    avgX = positions.reduce((s,p) => s + p.x, 0) / positions.length;
    avgY = positions.reduce((s,p) => s + p.y, 0) / positions.length;
  }
  const posLabel = avgX < 0.33 ? 'left' : avgX > 0.66 ? 'right' : 'center';
  const posLabelY = avgY < 0.33 ? 'top' : avgY > 0.66 ? 'bottom' : 'middle';

  // First and last seen
  const firstSeen = sightings[0]?.ts || '—';
  const lastSeen = sightings[sightings.length-1]?.ts || '—';

  let html = `
    <div class="person-header">
      ${latestImg ? `<img class="person-header-avatar" src="${latestImg}" onerror="this.style.display='none'">` : ''}
      <div class="person-header-info">
        <h2>${selectedUid}</h2>
        <p>${sightings.length} sightings over ${uniqueDays} day${uniqueDays!==1?'s':''}</p>
      </div>
    </div>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-card-title">total sightings</div>
        <div class="stat-card-value">${sightings.length}</div>
        <div class="stat-card-sub">First: ${firstSeen}<br>Last: ${lastSeen}</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">peak activity</div>
        <div class="stat-card-value">${String(peakHour).padStart(2,'0')}:00</div>
        <div class="stat-card-sub">Most active hour · Peak day: ${peakDay}</div>
      </div>

      <div class="stat-card" style="grid-column:1/3">
        <div class="stat-card-title">hourly activity</div>
        <div class="hour-chart">${hours.map((count,h) =>
          `<div class="hour-bar-wrap"><div class="hour-bar${h===peakHour?' peak':''}" style="height:${(count/maxHour)*100}%"></div>${h%3===0?`<div class="hour-label">${String(h).padStart(2,'0')}</div>`:''}</div>`
        ).join('')}</div>
      </div>

      <div class="stat-card">
        <div class="stat-card-title">day of week</div>
        <div class="day-chart">${dayNames.map((name,i) =>
          `<div class="day-bar-wrap"><div class="day-bar"><div class="day-bar-fill" style="height:${(days[i]/maxDay)*100}%"></div></div><div class="day-label">${name}</div><div class="day-val">${days[i]}</div></div>`
        ).join('')}</div>
      </div>

      <div class="stat-card">
        <div class="stat-card-title">camera position heatmap</div>
        <div class="heatmap-container" id="heatmap-${selectedUid}">
          <div class="heatmap-axis" style="top:4px;left:50%;">top</div>
          <div class="heatmap-axis" style="bottom:4px;left:50%;">bottom</div>
          <div class="heatmap-axis" style="left:4px;top:50%;">L</div>
          <div class="heatmap-axis" style="right:4px;top:50%;">R</div>
        </div>
        <div class="stat-card-sub" style="margin-top:8px">
          Strongest zone: <strong>${posLabel}-${posLabelY}</strong> ·
          Avg position: (${(avgX*100).toFixed(0)}%, ${(avgY*100).toFixed(0)}%)
        </div>
      </div>
    </div>
  `;
  main.innerHTML = html;

  // Render heatmap dots
  const hm = document.getElementById('heatmap-' + selectedUid);
  if (hm && positions.length > 0) {
    // Sample up to 300 positions for performance
    const sample = positions.length > 300
      ? positions.filter((_,i) => i % Math.ceil(positions.length/300) === 0)
      : positions;
    sample.forEach(p => {
      const dot = document.createElement('div');
      dot.className = 'heatmap-dot';
      dot.style.left = (p.x * 100) + '%';
      dot.style.top = (p.y * 100) + '%';
      hm.appendChild(dot);
    });
    // Center marker
    const c = document.createElement('div');
    c.className = 'heatmap-center';
    c.style.left = (avgX*100)+'%'; c.style.top = (avgY*100)+'%';
    hm.appendChild(c);
    const lbl = document.createElement('div');
    lbl.className = 'heatmap-label';
    lbl.style.left = (avgX*100)+'%'; lbl.style.top = (avgY*100)+'%';
    lbl.textContent = 'avg';
    hm.appendChild(lbl);
  }
}

function openLightbox(src, uid, ts) { event.stopPropagation(); document.getElementById('lightbox-img').src=src; document.getElementById('lightbox-meta').innerHTML=`<span>${uid}</span> — ${ts}`; document.getElementById('lightbox').classList.add('open'); }
function closeLightbox() { document.getElementById('lightbox').classList.remove('open'); }
document.addEventListener('keydown', e => { if(e.key==='Escape') closeLightbox(); });

loadAll();
setInterval(loadAll, 5000);
</script>
</body>
</html>
"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, snapshot_dir=DEFAULT_DIR, stats_path=DEFAULT_STATS, **kwargs):
        self.snapshot_dir = snapshot_dir
        self.stats_path = stats_path
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ('/', '/index.html'):
            self._html()
        elif path == '/api/snapshots':
            self._json(get_snapshot_data(self.snapshot_dir))
        elif path == '/api/stats':
            self._json(get_stats_data(self.stats_path))
        elif path.startswith('/snapshots/'):
            self._snapshot(path)
        else:
            self.send_error(404)

    def _html(self):
        c = HTML_PAGE.encode()
        self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',len(c)); self.end_headers(); self.wfile.write(c)

    def _json(self, data):
        c = json.dumps(data).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',len(c)); self.send_header('Cache-Control','no-cache'); self.end_headers(); self.wfile.write(c)

    def _snapshot(self, path):
        rel = path.replace('/snapshots/','',1)
        fp = os.path.normpath(os.path.join(self.snapshot_dir, rel))
        if not fp.startswith(os.path.normpath(self.snapshot_dir)) or not os.path.isfile(fp):
            self.send_error(404); return
        ext = os.path.splitext(fp)[1].lower().lstrip('.')
        mime = {'jpg':'image/jpeg','jpeg':'image/jpeg','png':'image/png'}.get(ext,'application/octet-stream')
        with open(fp,'rb') as f: c = f.read()
        self.send_response(200); self.send_header('Content-Type',mime); self.send_header('Content-Length',len(c)); self.send_header('Cache-Control','max-age=60'); self.end_headers(); self.wfile.write(c)

    def log_message(self, fmt, *args):
        if '404' in str(args) or '500' in str(args): super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="Face Tracker Dashboard")
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--dir', default=DEFAULT_DIR, help='Snapshot directory')
    parser.add_argument('--stats', default=DEFAULT_STATS, help='Stats JSON file')
    args = parser.parse_args()

    snap_dir = os.path.abspath(args.dir)
    stats_path = os.path.abspath(args.stats)

    def handler(*a, **kw):
        return Handler(*a, snapshot_dir=snap_dir, stats_path=stats_path, **kw)

    server = HTTPServer(('0.0.0.0', args.port), handler)
    print("=" * 48)
    print("  Face Tracker Dashboard")
    print("=" * 48)
    print(f"  Snapshots:   {snap_dir}")
    print(f"  Stats:       {stats_path}")
    print(f"  Server:      http://localhost:{args.port}")
    print("=" * 48)
    print(f"\nOpen http://localhost:{args.port} in your browser.")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == '__main__':
    main()
