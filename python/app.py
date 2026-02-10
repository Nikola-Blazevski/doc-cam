#!/usr/bin/env python3
"""
Face Tracker — Master Dashboard

Single application that runs camera pipelines + web UI.
Live camera streams are served as MJPEG over HTTP.

Usage:
    python app.py 0                         # single webcam
    python app.py 0 1                       # two cameras
    python app.py 0 1 --mode1 rgb --mode2 ir
    python app.py 0 --port 8080             # custom port

Then open http://localhost:8080 in your browser.
"""

# Suppress noisy warnings before any imports
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import argparse
import json
import sys
import time
import threading
import traceback
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import urllib.parse

import numpy as np
import cv2 as cv

from config import TARGET_FPS, FRAME_DELAY, FACE_DB_PATH, FACE_SNAPSHOT_DIR, STATS_DB_PATH
from gpu_utils import check_gpu, print_gpu_status
from face_tracker import FaceTracker
from pipeline import CameraPipeline


# =====================================================================
# Global state shared between camera threads and HTTP server
# =====================================================================

class AppState:
    """Holds all shared state for the application."""

    def __init__(self):
        self.tracker = None
        self.gpu_info = {}
        self.pipelines = []       # list of CameraPipeline
        self.cam_labels = []      # list of str labels
        self.cam_enabled = []     # list of bool (toggle on/off)
        self.cam_show_processed = []  # list of bool (raw vs processed)

        # Latest JPEG-encoded frames per camera
        self._frame_lock = threading.Lock()
        self._latest_frames = {}  # idx -> bytes (JPEG)

        self.running = True
        self.fps = TARGET_FPS

    def set_frame(self, idx, jpeg_bytes):
        with self._frame_lock:
            self._latest_frames[idx] = jpeg_bytes

    def get_frame(self, idx):
        with self._frame_lock:
            return self._latest_frames.get(idx)

    def get_status(self):
        """Return JSON-serializable status dict."""
        cams = []
        for i, p in enumerate(self.pipelines):
            cams.append({
                "index": i,
                "label": self.cam_labels[i],
                "enabled": self.cam_enabled[i],
                "show_processed": self.cam_show_processed[i],
                "healthy": p.is_healthy,
                "fps": p.display_fps,
                "mode": p.mode,
                "source": str(p.source),
            })
        return {
            "cameras": cams,
            "gpu": {
                "name": self.gpu_info.get("gpu_name", "None"),
                "memory": self.gpu_info.get("gpu_memory", "N/A"),
                "opencv_cuda": self.gpu_info.get("opencv_cuda", False),
                "dlib_cuda": self.gpu_info.get("dlib_cuda", False),
            },
            "identities": self.tracker.get_total_known() if self.tracker else 0,
            "active_faces": self.tracker.get_person_count() if self.tracker else 0,
        }


STATE = AppState()


# =====================================================================
# Camera processing thread
# =====================================================================

def camera_loop(idx):
    """Background thread: read frames, process, encode to JPEG."""
    pipeline = STATE.pipelines[idx]
    while STATE.running:
        try:
            if not STATE.cam_enabled[idx]:
                time.sleep(0.1)
                continue

            if not pipeline.is_healthy:
                time.sleep(0.5)
                continue

            ret, img = pipeline.read_frame()
            if not ret or img is None:
                time.sleep(0.05)
                continue

            orig, proc = pipeline.process(img)
            frame = proc if STATE.cam_show_processed[idx] else orig

            # Encode to JPEG
            _, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 75])
            STATE.set_frame(idx, buf.tobytes())

            # Pace to target FPS
            time.sleep(FRAME_DELAY)

        except Exception as e:
            print(f"  [Camera {idx}] Loop error: {e}")
            time.sleep(0.2)


# =====================================================================
# Snapshot + stats data helpers
# =====================================================================

def get_snapshot_data():
    base = Path(FACE_SNAPSHOT_DIR)
    if not base.exists():
        return {"people": []}
    people = []
    for uid_dir in sorted(base.iterdir()):
        if not uid_dir.is_dir():
            continue
        uid = uid_dir.name
        photos = []
        for f in sorted(uid_dir.iterdir(), reverse=True):
            if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            stem = f.stem
            try:
                dt = datetime.strptime(stem[:15], "%Y%m%d_%H%M%S")
                ms = stem[16:] if len(stem) > 15 else "000"
                ts = dt.strftime("%b %d, %Y  %H:%M:%S") + f".{ms}"
            except (ValueError, IndexError):
                ts = stem
            photos.append({"filename": f.name, "path": f"/snapshots/{uid}/{f.name}", "timestamp": ts})
        if photos:
            people.append({"uid": uid, "photo_count": len(photos), "photos": photos, "latest": photos[0]["path"]})
    people.sort(key=lambda p: p["photo_count"], reverse=True)
    return {"people": people}


def get_stats_data():
    if not os.path.exists(STATS_DB_PATH):
        return {}
    try:
        with open(STATS_DB_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


# =====================================================================
# HTTP Handler
# =====================================================================

HTML_PAGE = None  # Loaded below

class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))

        if path == '/' or path == '/index.html':
            self._serve_bytes(HTML_PAGE.encode(), 'text/html; charset=utf-8')
        elif path == '/api/status':
            self._serve_json(STATE.get_status())
        elif path == '/api/snapshots':
            self._serve_json(get_snapshot_data())
        elif path == '/api/stats':
            self._serve_json(get_stats_data())
        elif path.startswith('/api/stream/'):
            self._serve_mjpeg(path)
        elif path.startswith('/api/toggle/'):
            self._handle_toggle(path)
        elif path.startswith('/api/view/'):
            self._handle_view_toggle(path)
        elif path.startswith('/snapshots/'):
            self._serve_snapshot(path)
        else:
            self.send_error(404)

    def do_POST(self):
        self.do_GET()

    def _serve_bytes(self, data, content_type, cache='no-cache'):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(data))
        self.send_header('Cache-Control', cache)
        self.end_headers()
        self.wfile.write(data)

    def _serve_json(self, obj):
        self._serve_bytes(json.dumps(obj).encode(), 'application/json')

    def _serve_mjpeg(self, path):
        """Stream MJPEG for a camera index."""
        try:
            idx = int(path.split('/')[-1])
        except (ValueError, IndexError):
            self.send_error(400)
            return
        if idx < 0 or idx >= len(STATE.pipelines):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()

        try:
            while STATE.running:
                jpeg = STATE.get_frame(idx)
                if jpeg:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                time.sleep(FRAME_DELAY)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_toggle(self, path):
        """Toggle camera on/off."""
        try:
            idx = int(path.split('/')[-1])
            if 0 <= idx < len(STATE.cam_enabled):
                STATE.cam_enabled[idx] = not STATE.cam_enabled[idx]
                self._serve_json({"index": idx, "enabled": STATE.cam_enabled[idx]})
            else:
                self.send_error(404)
        except Exception:
            self.send_error(400)

    def _handle_view_toggle(self, path):
        """Toggle raw/processed view for a camera."""
        try:
            idx = int(path.split('/')[-1])
            if 0 <= idx < len(STATE.cam_show_processed):
                STATE.cam_show_processed[idx] = not STATE.cam_show_processed[idx]
                self._serve_json({"index": idx, "show_processed": STATE.cam_show_processed[idx]})
            else:
                self.send_error(404)
        except Exception:
            self.send_error(400)

    def _serve_snapshot(self, path):
        rel = path.replace('/snapshots/', '', 1)
        fp = os.path.normpath(os.path.join(FACE_SNAPSHOT_DIR, rel))
        norm_base = os.path.normpath(FACE_SNAPSHOT_DIR)
        if not fp.startswith(norm_base) or not os.path.isfile(fp):
            self.send_error(404)
            return
        ext = os.path.splitext(fp)[1].lower().lstrip('.')
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext, 'application/octet-stream')
        with open(fp, 'rb') as f:
            data = f.read()
        self._serve_bytes(data, mime, 'max-age=60')

    def log_message(self, fmt, *args):
        if '404' in str(args) or '500' in str(args):
            super().log_message(fmt, *args)


# =====================================================================
# HTML (embedded single-page app)
# =====================================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Tracker — Command Center</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
:root {
  --bg-0:#07070a; --bg-1:#0d0d11; --bg-2:#131318; --bg-3:#1a1a21; --bg-4:#222230;
  --border:#2a2a35; --border-hi:#3d3d4d;
  --t1:#ececf1; --t2:#a9a9b8; --t3:#6e6e80;
  --accent:#f59e0b; --accent-dim:#92400e; --accent-glow:rgba(245,158,11,.1);
  --green:#22c55e; --red:#ef4444; --blue:#3b82f6;
  --mono:'JetBrains Mono',monospace; --sans:'DM Sans',sans-serif;
  --r:6px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg-0);color:var(--t1);font-family:var(--sans);overflow:hidden;height:100vh}

/* ---- Top Bar ---- */
.topbar{height:48px;background:var(--bg-1);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 20px;gap:16px;z-index:100}
.topbar-logo{width:26px;height:26px;background:linear-gradient(135deg,var(--accent),var(--accent-dim));border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:12px;color:#000;font-weight:700}
.topbar-title{font-family:var(--mono);font-size:13px;font-weight:600} .topbar-title b{color:var(--accent)}
.topbar-stats{margin-left:auto;display:flex;gap:18px;font-family:var(--mono);font-size:10px;color:var(--t3)}
.topbar-stats .v{color:var(--accent);font-weight:600}
.topbar-gpu{font-family:var(--mono);font-size:10px;color:var(--t3);padding:3px 10px;border:1px solid var(--border);border-radius:var(--r)}

/* ---- Nav Tabs ---- */
.nav{display:flex;gap:2px;padding:0 20px;background:var(--bg-1);border-bottom:1px solid var(--border)}
.nav-tab{padding:10px 18px;font-family:var(--mono);font-size:11px;font-weight:500;color:var(--t3);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.nav-tab:hover{color:var(--t2)}
.nav-tab.active{color:var(--accent);border-bottom-color:var(--accent)}

/* ---- Page Container ---- */
.page-wrap{height:calc(100vh - 48px - 40px);overflow:hidden}
.page{display:none;height:100%;overflow-y:auto}
.page.active{display:flex;flex-direction:column}

/* ======== LIVE PAGE ======== */
.live-page{padding:16px;gap:16px}
.cam-controls{display:flex;gap:8px;flex-wrap:wrap}
.cam-chip{display:flex;align-items:center;gap:8px;padding:6px 14px;background:var(--bg-2);border:1px solid var(--border);border-radius:var(--r);font-family:var(--mono);font-size:11px;color:var(--t2);cursor:default}
.cam-dot{width:8px;height:8px;border-radius:50%}
.cam-dot.on{background:var(--green)} .cam-dot.off{background:var(--red)}
.cam-toggle{background:none;border:1px solid var(--border);color:var(--t2);padding:2px 10px;border-radius:3px;font-family:var(--mono);font-size:10px;cursor:pointer;transition:all .12s}
.cam-toggle:hover{border-color:var(--accent);color:var(--accent)}
.cam-toggle.active-btn{border-color:var(--green);color:var(--green)}
.cam-view-btn{background:none;border:1px solid var(--border);color:var(--t3);padding:2px 8px;border-radius:3px;font-family:var(--mono);font-size:9px;cursor:pointer;transition:all .12s}
.cam-view-btn:hover{border-color:var(--blue);color:var(--blue)}
.cam-view-btn.proc{border-color:var(--blue);color:var(--blue)}

.cam-grid{flex:1;display:grid;gap:12px;min-height:0}
.cam-grid.cols-1{grid-template-columns:1fr}
.cam-grid.cols-2{grid-template-columns:1fr 1fr}
.cam-card{background:var(--bg-2);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;display:flex;flex-direction:column;min-height:0}
.cam-card-head{display:flex;align-items:center;gap:8px;padding:6px 10px;border-bottom:1px solid var(--border);background:var(--bg-1)}
.cam-card-title{font-family:var(--mono);font-size:10px;color:var(--t2);flex:1}
.cam-card-fps{font-family:var(--mono);font-size:10px;color:var(--accent)}
.cam-card img{width:100%;height:100%;object-fit:contain;background:#000;display:block;min-height:0;flex:1}
.cam-off{display:flex;align-items:center;justify-content:center;flex:1;color:var(--t3);font-family:var(--mono);font-size:12px;background:var(--bg-0)}

/* ======== GALLERY PAGE ======== */
.gallery-page{flex-direction:row !important}
.g-sidebar{width:240px;min-width:240px;border-right:1px solid var(--border);overflow-y:auto;background:var(--bg-1)}
.g-sidebar-title{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:var(--t3);padding:12px 12px 6px}
.g-person{display:flex;align-items:center;gap:8px;padding:7px 12px;cursor:pointer;border-left:3px solid transparent;transition:all .1s}
.g-person:hover{background:var(--bg-3)}
.g-person.active{background:var(--bg-3);border-left-color:var(--accent)}
.g-avatar{width:32px;height:32px;border-radius:50%;object-fit:cover;border:2px solid var(--border);background:var(--bg-2)}
.g-person.active .g-avatar{border-color:var(--accent)}
.g-uid{font-family:var(--mono);font-size:11px;font-weight:600}
.g-person.active .g-uid{color:var(--accent)}
.g-sub{font-size:10px;color:var(--t3)}
.g-main{flex:1;overflow-y:auto;padding:16px}
.g-header{display:flex;align-items:center;gap:12px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)}
.g-header-av{width:44px;height:44px;border-radius:50%;object-fit:cover;border:2px solid var(--accent);background:var(--bg-2)}
.g-header h2{font-family:var(--mono);font-size:16px;color:var(--accent)}
.g-header p{font-size:11px;color:var(--t3)}
.photo-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
.p-card{background:var(--bg-2);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;cursor:pointer;transition:all .12s}
.p-card:hover{border-color:var(--accent);transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,.3)}
.p-card img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:var(--bg-0)}
.p-card-ts{padding:4px 6px;font-family:var(--mono);font-size:8px;color:var(--t3);border-top:1px solid var(--border)}
.empty{display:flex;align-items:center;justify-content:center;height:100%;color:var(--t3);font-family:var(--mono);font-size:12px}

/* ======== STATS PAGE ======== */
.stats-page{flex-direction:row !important}
.s-main{flex:1;overflow-y:auto;padding:16px}
.stats-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:900px){.stats-row{grid-template-columns:1fr}}
.s-card{background:var(--bg-2);border:1px solid var(--border);border-radius:var(--r);padding:14px}
.s-card-title{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:.08em;color:var(--t3);margin-bottom:10px}
.s-big{font-family:var(--mono);font-size:26px;font-weight:700;color:var(--accent)}
.s-sub{font-size:10px;color:var(--t3);margin-top:4px}
.hour-chart{display:flex;align-items:flex-end;gap:1px;height:100px;padding-top:6px}
.hb-w{flex:1;display:flex;flex-direction:column;align-items:center;gap:1px;height:100%;justify-content:flex-end}
.hb{width:100%;background:var(--accent);border-radius:1px 1px 0 0;min-height:1px;opacity:.6;transition:height .3s}
.hb.pk{opacity:1}
.hb-l{font-family:var(--mono);font-size:7px;color:var(--t3)}
.day-row{display:flex;gap:4px}
.dw{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px}
.d-bar{width:100%;height:36px;background:var(--bg-0);border-radius:3px;position:relative;overflow:hidden}
.d-fill{position:absolute;bottom:0;width:100%;background:var(--accent);border-radius:3px;opacity:.5;transition:height .3s}
.d-l{font-family:var(--mono);font-size:8px;color:var(--t3)}
.d-v{font-family:var(--mono);font-size:8px;color:var(--t2)}
.hm-box{position:relative;width:100%;aspect-ratio:16/9;background:var(--bg-0);border-radius:var(--r);overflow:hidden;border:1px solid var(--border)}
.hm-dot{position:absolute;width:6px;height:6px;border-radius:50%;background:var(--accent);opacity:.12;transform:translate(-50%,-50%)}
.hm-ctr{position:absolute;width:14px;height:14px;border-radius:50%;border:2px solid var(--accent);background:rgba(245,158,11,.25);transform:translate(-50%,-50%)}
.hm-lbl{position:absolute;font-family:var(--mono);font-size:8px;color:var(--accent);transform:translate(-50%,10px)}
.hm-ax{position:absolute;font-family:var(--mono);font-size:7px;color:var(--t3)}

/* Lightbox */
.lb{display:none;position:fixed;inset:0;z-index:300;background:rgba(7,7,10,.93);backdrop-filter:blur(16px);align-items:center;justify-content:center;flex-direction:column;gap:8px;cursor:pointer}
.lb.open{display:flex}
.lb img{max-width:80vw;max-height:75vh;border-radius:var(--r);border:1px solid var(--border)}
.lb-meta{font-family:var(--mono);font-size:11px;color:var(--t2)} .lb-meta b{color:var(--accent)}
.lb-x{position:absolute;top:14px;right:18px;font-size:18px;color:var(--t3);cursor:pointer;width:30px;height:30px;display:flex;align-items:center;justify-content:center;border-radius:50%;border:1px solid var(--border);transition:all .12s}
.lb-x:hover{color:var(--accent);border-color:var(--accent)}

::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-track{background:transparent} ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-logo">⬡</div>
  <div class="topbar-title">face<b>tracker</b></div>
  <div class="topbar-gpu" id="gpu-chip">—</div>
  <div class="topbar-stats">
    <span>identities <span class="v" id="st-id">0</span></span>
    <span>active <span class="v" id="st-active">0</span></span>
    <span>sightings <span class="v" id="st-sight">0</span></span>
  </div>
</div>

<div class="nav">
  <div class="nav-tab active" data-tab="live" onclick="switchTab('live')">◉ live</div>
  <div class="nav-tab" data-tab="gallery" onclick="switchTab('gallery')">◫ gallery</div>
  <div class="nav-tab" data-tab="stats" onclick="switchTab('stats')">◈ stats</div>
</div>

<div class="page-wrap">

<!-- ============ LIVE ============ -->
<div class="page active" id="page-live">
  <div class="live-page">
    <div class="cam-controls" id="cam-controls"></div>
    <div class="cam-grid" id="cam-grid"></div>
  </div>
</div>

<!-- ============ GALLERY ============ -->
<div class="page" id="page-gallery">
  <div class="gallery-page">
    <div class="g-sidebar"><div class="g-sidebar-title">persons</div><div id="g-list"></div></div>
    <div class="g-main" id="g-main"><div class="empty">select a person</div></div>
  </div>
</div>

<!-- ============ STATS ============ -->
<div class="page" id="page-stats">
  <div class="stats-page">
    <div class="g-sidebar"><div class="g-sidebar-title">persons</div><div id="s-list"></div></div>
    <div class="s-main" id="s-main"><div class="empty">select a person</div></div>
  </div>
</div>

</div>

<div class="lb" id="lb" onclick="closeLB()"><div class="lb-x" onclick="closeLB()">✕</div><img id="lb-img" src=""><div class="lb-meta" id="lb-meta"></div></div>

<script>
let STATUS={cameras:[]}, SNAPS={people:[]}, STATS={};
let selGallery=null, selStats=null;

/* ---- Tab switching ---- */
function switchTab(tab) {
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===tab));
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+tab).classList.add('active');
}

/* ---- Data loading ---- */
async function loadStatus(){ try{ STATUS=await(await fetch('/api/status')).json(); renderLive(); updateTopbar(); }catch(e){} }
async function loadSnaps(){ try{ SNAPS=await(await fetch('/api/snapshots')).json(); renderGallerySidebar(); }catch(e){} }
async function loadStats(){ try{ STATS=await(await fetch('/api/stats')).json(); renderStatsSidebar(); updateTopbar(); }catch(e){} }
async function loadAll(){ await Promise.all([loadStatus(),loadSnaps(),loadStats()]); }

function updateTopbar(){
  document.getElementById('st-id').textContent=STATUS.identities||0;
  document.getElementById('st-active').textContent=STATUS.active_faces||0;
  let ts=0; for(const u in STATS) ts+=(STATS[u].sightings||[]).length;
  document.getElementById('st-sight').textContent=ts;
  const g=STATUS.gpu||{};
  document.getElementById('gpu-chip').textContent=g.name!=='None'?`${g.name} (${g.memory})`:'CPU only';
}

/* ---- LIVE ---- */
function renderLive(){
  const cams=STATUS.cameras||[];
  const ctrl=document.getElementById('cam-controls');
  ctrl.innerHTML='';
  cams.forEach(c=>{
    const d=document.createElement('div');d.className='cam-chip';
    d.innerHTML=`<div class="cam-dot ${c.enabled?'on':'off'}"></div>
      <span>${c.label}</span>
      <button class="cam-toggle ${c.enabled?'active-btn':''}" onclick="toggleCam(${c.index})">${c.enabled?'ON':'OFF'}</button>
      <button class="cam-view-btn ${c.show_processed?'proc':''}" onclick="toggleView(${c.index})">${c.show_processed?'processed':'original'}</button>`;
    ctrl.appendChild(d);
  });

  const grid=document.getElementById('cam-grid');
  const enabledCams=cams.filter(c=>c.enabled);
  grid.className='cam-grid '+(enabledCams.length<=1?'cols-1':'cols-2');
  grid.innerHTML='';
  if(enabledCams.length===0){ grid.innerHTML='<div class="empty">all cameras disabled</div>'; return; }
  enabledCams.forEach(c=>{
    const card=document.createElement('div');card.className='cam-card';
    card.innerHTML=`<div class="cam-card-head"><div class="cam-card-title">${c.label}</div><div class="cam-card-fps">${c.fps} fps · ${c.mode.toUpperCase()}</div></div>
      <img src="/api/stream/${c.index}" alt="${c.label}">`;
    grid.appendChild(card);
  });
}

async function toggleCam(idx){ await fetch('/api/toggle/'+idx); loadStatus(); }
async function toggleView(idx){ await fetch('/api/view/'+idx); loadStatus(); }

/* ---- GALLERY ---- */
function renderGallerySidebar(){
  const list=document.getElementById('g-list');list.innerHTML='';
  (SNAPS.people||[]).forEach(p=>{
    const d=document.createElement('div');d.className='g-person'+(p.uid===selGallery?' active':'');
    d.onclick=()=>{selGallery=p.uid;renderGallerySidebar();renderGalleryMain()};
    d.innerHTML=`<img class="g-avatar" src="${p.latest}" onerror="this.style.visibility='hidden'"><div><div class="g-uid">${p.uid}</div><div class="g-sub">${p.photo_count} photos</div></div>`;
    list.appendChild(d);
  });
}
function renderGalleryMain(){
  const m=document.getElementById('g-main');
  const p=(SNAPS.people||[]).find(x=>x.uid===selGallery);
  if(!p){m.innerHTML='<div class="empty">no photos</div>';return;}
  let h=`<div class="g-header"><img class="g-header-av" src="${p.latest}" onerror="this.style.display='none'"><div><h2>${p.uid}</h2><p>${p.photo_count} captures</p></div></div><div class="photo-grid">`;
  p.photos.forEach(ph=>{h+=`<div class="p-card" onclick="openLB('${ph.path}','${p.uid}','${ph.timestamp.replace(/'/g,"\\'")}')"><img src="${ph.path}" loading="lazy"><div class="p-card-ts">${ph.timestamp}</div></div>`;});
  m.innerHTML=h+'</div>';
}

/* ---- STATS ---- */
function renderStatsSidebar(){
  const allUids=new Set();
  (SNAPS.people||[]).forEach(p=>allUids.add(p.uid));
  for(const u in STATS) allUids.add(u);
  const items=[...allUids].map(uid=>{
    const s=STATS[uid]; const sn=(SNAPS.people||[]).find(p=>p.uid===uid);
    return {uid, sightings:s?(s.sightings||[]).length:0, latest:sn?sn.latest:null};
  }).sort((a,b)=>b.sightings-a.sightings);

  const list=document.getElementById('s-list');list.innerHTML='';
  items.forEach(it=>{
    const d=document.createElement('div');d.className='g-person'+(it.uid===selStats?' active':'');
    d.onclick=()=>{selStats=it.uid;renderStatsSidebar();renderStatsMain()};
    const av=it.latest?`<img class="g-avatar" src="${it.latest}" onerror="this.style.visibility='hidden'">`:`<div class="g-avatar"></div>`;
    d.innerHTML=`${av}<div><div class="g-uid">${it.uid}</div><div class="g-sub">${it.sightings} sightings</div></div>`;
    list.appendChild(d);
  });
}
function renderStatsMain(){
  const m=document.getElementById('s-main');
  const st=STATS[selStats];
  if(!st||!st.sightings||!st.sightings.length){m.innerHTML='<div class="empty">no stats yet</div>';return;}
  const sn=(SNAPS.people||[]).find(p=>p.uid===selStats);
  const S=st.sightings, P=st.positions||[];
  const hours=Array(24).fill(0); S.forEach(s=>hours[s.hour]++);
  const mxH=Math.max(...hours,1), pkH=hours.indexOf(mxH);
  const days=Array(7).fill(0); S.forEach(s=>{if(s.weekday!==undefined)days[s.weekday]++});
  const mxD=Math.max(...days,1); const DN=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const pkD=DN[days.indexOf(Math.max(...days))];
  const uniqD=new Set(S.map(s=>s.day)).size;
  let ax=.5,ay=.5;
  if(P.length){ax=P.reduce((s,p)=>s+p.x,0)/P.length;ay=P.reduce((s,p)=>s+p.y,0)/P.length}
  const zx=ax<.33?'left':ax>.66?'right':'center', zy=ay<.33?'top':ay>.66?'bottom':'middle';
  const first=S[0]?.ts||'—', last=S[S.length-1]?.ts||'—';

  let h=`<div class="g-header">${sn?`<img class="g-header-av" src="${sn.latest}" onerror="this.style.display='none'">`:''}
    <div><h2>${selStats}</h2><p>${S.length} sightings · ${uniqD} day${uniqD!==1?'s':''}</p></div></div>
    <div class="stats-row">
      <div class="s-card"><div class="s-card-title">total sightings</div><div class="s-big">${S.length}</div><div class="s-sub">First: ${first}<br>Last: ${last}</div></div>
      <div class="s-card"><div class="s-card-title">peak hour</div><div class="s-big">${String(pkH).padStart(2,'0')}:00</div><div class="s-sub">Peak day: ${pkD}</div></div>
      <div class="s-card" style="grid-column:1/3"><div class="s-card-title">hourly activity</div>
        <div class="hour-chart">${hours.map((c,i)=>`<div class="hb-w"><div class="hb${i===pkH?' pk':''}" style="height:${(c/mxH)*100}%"></div>${i%3===0?`<div class="hb-l">${String(i).padStart(2,'0')}</div>`:''}</div>`).join('')}</div></div>
      <div class="s-card"><div class="s-card-title">day of week</div>
        <div class="day-row">${DN.map((n,i)=>`<div class="dw"><div class="d-bar"><div class="d-fill" style="height:${(days[i]/mxD)*100}%"></div></div><div class="d-l">${n}</div><div class="d-v">${days[i]}</div></div>`).join('')}</div></div>
      <div class="s-card"><div class="s-card-title">position heatmap</div>
        <div class="hm-box" id="hm-${selStats}">
          <div class="hm-ax" style="top:2px;left:50%">top</div><div class="hm-ax" style="bottom:2px;left:50%">bottom</div>
          <div class="hm-ax" style="left:3px;top:50%">L</div><div class="hm-ax" style="right:3px;top:50%">R</div>
        </div>
        <div class="s-sub" style="margin-top:6px">Zone: <b>${zx}-${zy}</b> · Avg: (${(ax*100).toFixed(0)}%, ${(ay*100).toFixed(0)}%)</div></div>
    </div>`;
  m.innerHTML=h;
  // Heatmap dots
  const hm=document.getElementById('hm-'+selStats);
  if(hm&&P.length){
    const sample=P.length>250?P.filter((_,i)=>i%Math.ceil(P.length/250)===0):P;
    sample.forEach(p=>{const d=document.createElement('div');d.className='hm-dot';d.style.left=(p.x*100)+'%';d.style.top=(p.y*100)+'%';hm.appendChild(d)});
    const c=document.createElement('div');c.className='hm-ctr';c.style.left=(ax*100)+'%';c.style.top=(ay*100)+'%';hm.appendChild(c);
    const l=document.createElement('div');l.className='hm-lbl';l.style.left=(ax*100)+'%';l.style.top=(ay*100)+'%';l.textContent='avg';hm.appendChild(l);
  }
}

/* ---- Lightbox ---- */
function openLB(src,uid,ts){event.stopPropagation();document.getElementById('lb-img').src=src;document.getElementById('lb-meta').innerHTML=`<b>${uid}</b> — ${ts}`;document.getElementById('lb').classList.add('open')}
function closeLB(){document.getElementById('lb').classList.remove('open')}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLB()});

/* ---- Init ---- */
loadAll();
setInterval(()=>{loadStatus();loadStats()},3000);
setInterval(loadSnaps,8000);
</script>
</body>
</html>"""


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Face Tracker — Master Dashboard")
    parser.add_argument("source1", help="First camera index or video path")
    parser.add_argument("source2", nargs="?", default=None, help="Second camera (optional)")
    parser.add_argument("--mode1", default="rgb", choices=["rgb", "ir"])
    parser.add_argument("--mode2", default="ir", choices=["rgb", "ir"])
    parser.add_argument("--db", default=FACE_DB_PATH)
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    parser.add_argument("--port", type=int, default=8080, help="Web UI port")
    args = parser.parse_args()
    try:
        args.source1 = int(args.source1) if args.source1.isdigit() else args.source1
    except (AttributeError, ValueError):
        pass
    if args.source2 is not None:
        try:
            args.source2 = int(args.source2) if args.source2.isdigit() else args.source2
        except (AttributeError, ValueError):
            pass
    return args


def main():
    args = parse_args()

    # GPU
    gpu_info = check_gpu()
    print_gpu_status(gpu_info)
    STATE.gpu_info = gpu_info

    # Tracker
    print(f"\nFace database: {args.db}")
    STATE.tracker = FaceTracker(use_cuda=gpu_info["dlib_cuda"], db_path=args.db)
    print(f"  Known identities: {STATE.tracker.get_total_known()}")

    # Cameras
    cameras = [(args.source1, args.mode1, "Cam 1")]
    if args.source2 is not None:
        cameras.append((args.source2, args.mode2, "Cam 2"))

    STATE.fps = args.fps
    print(f"\nInitializing {len(cameras)} camera(s)...")

    for source, mode, label in cameras:
        full_label = f"{label} [{mode.upper()}] (src:{source})"
        try:
            p = CameraPipeline(
                source=source, tracker=STATE.tracker,
                label=full_label, mode=mode,
                use_opencv_cuda=gpu_info["opencv_cuda"])
            STATE.pipelines.append(p)
            STATE.cam_labels.append(full_label)
            STATE.cam_enabled.append(True)
            STATE.cam_show_processed.append(False)
            print(f"  [{full_label}] Ready")
        except RuntimeError as e:
            print(f"  ERROR: {full_label}: {e}")
            for pp in STATE.pipelines:
                pp.close()
            sys.exit(1)

    # Start camera threads
    cam_threads = []
    for i in range(len(STATE.pipelines)):
        t = threading.Thread(target=camera_loop, args=(i,), daemon=True)
        t.start()
        cam_threads.append(t)

    # HTTP server (threaded)
    class ThreadedHTTPServer(HTTPServer):
        allow_reuse_address = True
        daemon_threads = True
        request_queue_size = 32

        def process_request(self, request, client_address):
            t = threading.Thread(target=self.process_request_thread, args=(request, client_address), daemon=True)
            t.start()

        def process_request_thread(self, request, client_address):
            try:
                self.finish_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
            finally:
                self.shutdown_request(request)

    server = ThreadedHTTPServer(('0.0.0.0', args.port), AppHandler)

    print(f"\n{'='*48}")
    print(f"  Face Tracker — Command Center")
    print(f"{'='*48}")
    print(f"  Cameras:  {len(cameras)}")
    print(f"  Web UI:   http://localhost:{args.port}")
    print(f"  FPS:      {args.fps}")
    print(f"{'='*48}")
    print(f"\nOpen http://localhost:{args.port} in your browser.")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        STATE.running = False
        try:
            STATE.tracker.save_db()
            STATE.tracker.stats.save()
        except Exception as e:
            print(f"  Save error: {e}")
        for p in STATE.pipelines:
            p.close()
        server.shutdown()
        print("Done.")


if __name__ == '__main__':
    main()
