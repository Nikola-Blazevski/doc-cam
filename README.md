# doc-cam - Hand & Face Detection with Persistent Identity Tracking

A real-time video processing pipeline that detects hands and faces, labels individual fingers (thumb, pointer, middle), and assigns persistent UUIDs to recognized faces across sessions.

** THIS PROJECT HAS A LOT OF AI CODE IN IT I CANNOT PROMISE ANY STABILITY FOR NOW**

## Project Structure

```
hand_face_tracker/
├── main.py              # Entry point — arg parsing, shared tracker, grid display
├── pipeline.py          # Per-camera pipeline (async face recognition)
├── config.py            # All constants and tunable parameters
├── gpu_utils.py         # GPU availability detection
├── detectors.py         # MediaPipe hand/face/pose detector factory
├── face_tracker.py      # Face identity — spatial matching, merging, snapshots
├── processing.py        # Image processing — RGB (skin-color) and IR (intensity)
├── drawing.py           # OpenCV overlays (hands, faces, body skeleton, HUD)
├── install.sh           # Full install script
├── download_models.sh   # Download all ML models
├── requirements.txt     # Python dependencies
├── known_faces.pkl      # (generated) Shared persistent face database
└── face_snapshots/      # (generated) Face crops organized by UUID
    ├── a1b2c3d4/
    │   ├── 20260209_143022_451.jpg
    │   ├── 20260209_143025_612.jpg
    │   └── ...
    └── e5f6g7h8/
        └── ...
```

## Setup

```bash
chmod +x install.sh
./install.sh              # CPU
./install.sh --gpu        # with CUDA support
```

## Usage

```bash
source venv/bin/activate

# Single RGB webcam
python main.py 0

# Two cameras: cam 0 = RGB, cam 1 = IR (default)
python main.py 0 1

# Explicit modes
python main.py 0 1 --mode1 rgb --mode2 ir

# Both RGB
python main.py 0 1 --mode1 rgb --mode2 rgb

# Custom FPS
python main.py 0 1 --fps 10

# Video files
python main.py video_rgb.mp4 video_ir.mp4
```

Each camera gets its own row in the window:
```
┌──────────────────────┬──────────────────────┐
│  Cam 1 [RGB] Original│  Cam 1 [RGB] Processed│
├──────────────────────┼──────────────────────┤
│  Cam 2 [IR] Original │  Cam 2 [IR] Processed │
└──────────────────────┴──────────────────────┘
```

Press **q** or **ESC** to quit.

## Face Identity Persistence

UUIDs are designed to be consistent across sessions and cameras:

- **Deterministic UUIDs**: derived from a hash of the face encoding
- **Spatial-aware matching**: combines encoding distance with bbox position so head movement doesn't cause UUID splits
- **UUID merging**: if two UIDs converge to the same person over time, they're automatically merged
- **Multi-sample matching**: stores up to 30 encoding samples per person
- **Shared database**: both cameras use a single `known_faces.pkl`
- **Auto-save**: database saves every 30 seconds
- **Persisted faces never expire**: only transient faces time out

## Face Snapshots

The system automatically saves face crops to `face_snapshots/<uuid>/`:

- One snapshot every **3 seconds** per person
- Timestamped filenames: `20260209_143022_451.jpg`
- 30% padding around the face bounding box
- Only faces ≥40px are saved (no tiny crops)
- Organized by UUID so each person gets their own folder

## Body Tracking

MediaPipe Pose Landmarker detects full body skeletons:

- Up to 4 bodies tracked simultaneously
- 33 landmarks per body (torso, arms, legs, face)
- Skeleton overlay with connection lines
- Bounding box around each detected body
- Visibility filtering (low-confidence landmarks hidden)
- Requires `pose_landmarker_lite.task` model (downloaded by install.sh)

## IR Camera Support

Standard skin-color (HSV) segmentation doesn't work on IR cameras because IR images lack color information. The IR pipeline instead uses:

1. **CLAHE** — normalizes uneven IR illumination
2. **Gaussian blur** — reduces IR sensor noise
3. **Otsu threshold** — automatically separates warm (bright skin) from cool (dark background)
4. **Morphological cleanup** — closes gaps and removes noise
5. **Contour filtering** — keeps only large warm blobs
6. **Adaptive threshold** — extracts fine detail from segmented regions

## GPU Acceleration

| Component | GPU Method | How to Enable |
|---|---|---|
| MediaPipe | OpenGL ES delegate | Use `mediapipe==0.10.31` on Ubuntu with OpenGL ES 3.1+ |
| OpenCV | CUDA kernels | Build OpenCV from source with CUDA |
| dlib | CUDA | `pip install dlib --config-settings="--build-option=--yes" --config-settings="--build-option=DLIB_USE_CUDA"` |

All GPU features auto-detect and fall back to CPU if unavailable.

## Configuration

Edit `config.py` to adjust:

- `TARGET_FPS` — frame rate cap (default: 5)
- `SKIN_LOWER` / `SKIN_UPPER` — HSV skin color range
- `FACE_MATCH_THRESHOLD` — how similar faces must be to match (lower = stricter)
- `FACE_EXPIRY_FRAMES` — frames before forgetting a missing face
- `FINGERS` — which fingers to highlight and their colors

## Multi-Camera Notes

- Each camera gets its own face database (`known_faces_cam1.pkl`, `known_faces_cam2.pkl`)
- Each camera runs its own independent detectors and trackers
- If cameras have different resolutions, frames are resized to match the first camera
- To share face identity across cameras, set both to the same `db_path` in `main.py`