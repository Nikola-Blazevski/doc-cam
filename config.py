"""Global configuration constants."""

import numpy as np

# --- FPS ---
TARGET_FPS = 60
FRAME_DELAY = 1.0 / TARGET_FPS

# --- Skin color thresholds (HSV) — for RGB cameras ---
SKIN_LOWER = np.array([0, 20, 70], dtype="uint8")
SKIN_UPPER = np.array([20, 255, 255], dtype="uint8")

# --- IR processing settings ---
IR_CLAHE_CLIP = 3.0
IR_CLAHE_GRID = (8, 8)
IR_BLUR_KSIZE = (5, 5)
IR_THRESH_BLOCK = 15
IR_THRESH_C = 3
IR_MORPH_KSIZE = 5
IR_MIN_CONTOUR_AREA = 500

# --- Hand skeleton connections ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

# --- Finger definitions: (name, tip_landmark_index, color_bgr) ---
FINGERS = [
    ("Thumb",   4,  (0, 200, 255)),
    ("Pointer", 8,  (255, 0, 0)),
    ("Middle",  12, (0, 0, 255)),
]

FINGER_CONNECTIONS = {
    "Thumb":   [(1, 2), (2, 3), (3, 4)],
    "Pointer": [(5, 6), (6, 7), (7, 8)],
    "Middle":  [(9, 10), (10, 11), (11, 12)],
}

FINGER_LANDMARKS = {
    "Thumb":   [1, 2, 3, 4],
    "Pointer": [5, 6, 7, 8],
    "Middle":  [9, 10, 11, 12],
}

# --- Face keypoint names ---
FACE_KEYPOINT_NAMES = ["L Eye", "R Eye", "Nose", "Mouth", "L Ear", "R Ear"]

# --- Face tracker settings ---
FACE_MATCH_THRESHOLD = 0.55         # encoding distance threshold
FACE_SPATIAL_WEIGHT = 0.15          # weight for spatial proximity in matching
FACE_EXPIRY_FRAMES = 150            # frames before forgetting in-session
FACE_DB_PATH = "known_faces.pkl"    # single shared DB
FACE_MAX_SAMPLES = 30               # encoding samples per person
FACE_AUTOSAVE_INTERVAL = 30         # seconds between auto-saves
FACE_MIN_SAMPLES_TO_PERSIST = 3     # sightings before saving
FACE_MERGE_THRESHOLD = 0.38         # if two UIDs are this close, merge them

# --- Face snapshot settings ---
FACE_SNAPSHOT_DIR = "face_snapshots"  # base directory
FACE_SNAPSHOT_INTERVAL = 3.0          # seconds between snapshots per person
FACE_SNAPSHOT_MIN_SIZE = 40           # minimum face crop size in pixels
FACE_SNAPSHOT_PAD = 0.3               # padding around face bbox (30%)

# --- Body pose settings ---
POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27),
    # Right leg
    (24, 26), (26, 28),
    # Face outline (shoulders to ears)
    (11, 0), (12, 0),
]
POSE_COLOR = (255, 180, 0)  # cyan-ish for body skeleton

# --- Model paths ---
HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"

# --- Stats ---
STATS_DB_PATH = "face_stats.json"
STATS_RECORD_INTERVAL = 1.0  # seconds between recording sightings per person

# --- Layout ---
LAYOUT = "grid"
