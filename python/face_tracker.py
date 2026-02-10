"""
Face identity tracking with robust cross-session UUID persistence.

Improvements for accuracy:
  - Spatial-aware matching: combines encoding distance with bbox position
    so a face that moved slightly doesn't get a new UUID
  - UUID merging: if two UIDs are found to be the same person (very close
    encodings), they are merged into one
  - Higher sample count for more stable mean encodings
  - Face snapshot saving: crops and saves face images per UUID
"""

import hashlib
import os
import pickle
import time
import threading
from datetime import datetime

import numpy as np
import cv2 as cv
import face_recognition

from config import (
    FACE_MATCH_THRESHOLD, FACE_EXPIRY_FRAMES, FACE_DB_PATH,
    FACE_MAX_SAMPLES, FACE_AUTOSAVE_INTERVAL, FACE_MIN_SAMPLES_TO_PERSIST,
    FACE_SPATIAL_WEIGHT, FACE_MERGE_THRESHOLD,
    FACE_SNAPSHOT_DIR, FACE_SNAPSHOT_INTERVAL, FACE_SNAPSHOT_MIN_SIZE,
    FACE_SNAPSHOT_PAD,
)
from stats_collector import StatsCollector


def _encoding_to_uuid(encoding):
    """Derive a deterministic 8-char hex ID from a face encoding."""
    try:
        quantized = np.round(encoding, decimals=1)
        raw = quantized.tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        return digest[:8]
    except Exception:
        import uuid
        return str(uuid.uuid4())[:8]


class FaceTracker:
    """
    Tracks faces across frames and sessions with persistent UUIDs.

    Uses spatial position + encoding distance for matching, and
    periodically merges UIDs that have converged to the same person.
    """

    def __init__(self, match_threshold=FACE_MATCH_THRESHOLD,
                 expiry_frames=FACE_EXPIRY_FRAMES,
                 db_path=FACE_DB_PATH,
                 use_cuda=False):
        self.match_threshold = match_threshold
        self.expiry_frames = expiry_frames
        self.db_path = db_path
        self.use_cuda = use_cuda
        self.frame_count = 0
        self.lock = threading.RLock()
        self.last_save_time = time.time()

        # Snapshot state: {uid: last_snapshot_time}
        self._snapshot_times = {}
        self._snapshot_dir = FACE_SNAPSHOT_DIR
        os.makedirs(self._snapshot_dir, exist_ok=True)

        # Stats collector
        self.stats = StatsCollector()

        self.known_faces = self._load_db()

    # ----- Persistence -----

    def _load_db(self):
        """Load known faces from disk."""
        if not os.path.exists(self.db_path):
            return {}
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                return {}

            cleaned = {}
            for uid, entry in data.items():
                try:
                    if not isinstance(entry, dict) or "encoding" not in entry:
                        continue
                    enc = np.asarray(entry["encoding"], dtype=np.float64)
                    if enc.shape != (128,):
                        continue
                    if "samples" not in entry:
                        entry["samples"] = [enc]
                    else:
                        valid = [np.asarray(s, dtype=np.float64)
                                 for s in entry["samples"]
                                 if np.asarray(s).shape == (128,)]
                        entry["samples"] = valid if valid else [enc]
                    entry["encoding"] = np.mean(entry["samples"], axis=0)
                    entry["persisted"] = True
                    entry["last_seen"] = 0
                    entry.setdefault("last_bbox_center", None)
                    cleaned[uid] = entry
                except Exception:
                    continue
            print(f"Loaded {len(cleaned)} known faces from {self.db_path}")
            return cleaned
        except Exception as e:
            print(f"Warning: could not load {self.db_path}: {e}")
            return {}

    def save_db(self):
        """Save faces with enough samples."""
        with self.lock:
            try:
                to_save = {}
                for uid, entry in self.known_faces.items():
                    samples = entry.get("samples", [])
                    if len(samples) >= FACE_MIN_SAMPLES_TO_PERSIST:
                        to_save[uid] = {
                            "encoding": entry["encoding"].copy(),
                            "samples": [s.copy() for s in samples[-FACE_MAX_SAMPLES:]],
                            "persisted": True,
                            "last_seen": 0,
                            "last_bbox_center": None,
                        }
                tmp_path = self.db_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    pickle.dump(to_save, f)
                os.replace(tmp_path, self.db_path)
                print(f"Saved {len(to_save)} known faces to {self.db_path}")
            except Exception as e:
                print(f"Warning: could not save face DB: {e}")
        self.last_save_time = time.time()

    # ----- Identification -----

    def identify(self, rgb_image, face_locations):
        """
        Identify faces. Returns list of UUID strings.

        Also saves face snapshots and triggers merging.
        """
        with self.lock:
            self.frame_count += 1

            if not face_locations:
                self._cleanup()
                return []

            h, w = rgb_image.shape[:2]
            valid_locations = []
            bbox_centers = []
            for loc in face_locations:
                try:
                    top, right, bottom, left = loc
                    top, right = max(0, int(top)), min(w, int(right))
                    bottom, left = min(h, int(bottom)), max(0, int(left))
                    if bottom - top < 10 or right - left < 10:
                        continue
                    valid_locations.append((top, right, bottom, left))
                    cx = (left + right) / 2.0 / w
                    cy = (top + bottom) / 2.0 / h
                    bbox_centers.append((cx, cy))
                except (TypeError, ValueError):
                    continue

            if not valid_locations:
                self._cleanup()
                return []

            try:
                model = "large" if self.use_cuda else "small"
                encodings = face_recognition.face_encodings(
                    rgb_image, valid_locations, model=model)
            except Exception as e:
                print(f"Warning: face encoding failed: {e}")
                self._cleanup()
                return ["???"] * len(valid_locations)

            ids = []
            used_uids = set()  # prevent two faces matching same UID

            for i, encoding in enumerate(encodings):
                try:
                    encoding = np.asarray(encoding, dtype=np.float64)
                    if encoding.shape != (128,) or not np.all(np.isfinite(encoding)):
                        ids.append("???")
                        continue

                    center = bbox_centers[i] if i < len(bbox_centers) else None
                    match_id = self._find_match(encoding, center, used_uids)

                    if match_id is None:
                        match_id = _encoding_to_uuid(encoding)
                        # Handle collision
                        if match_id in self.known_faces:
                            existing_dist = np.linalg.norm(
                                self.known_faces[match_id]["encoding"] - encoding)
                            if existing_dist > self.match_threshold:
                                suffix = _encoding_to_uuid(
                                    encoding + np.random.randn(128) * 0.01)
                                match_id = match_id[:6] + suffix[-2:]

                        self.known_faces[match_id] = {
                            "encoding": encoding.copy(),
                            "samples": [encoding.copy()],
                            "last_seen": self.frame_count,
                            "persisted": False,
                            "last_bbox_center": center,
                        }
                    else:
                        entry = self.known_faces[match_id]
                        samples = entry.get("samples", [entry["encoding"].copy()])
                        samples.append(encoding.copy())
                        if len(samples) > FACE_MAX_SAMPLES:
                            samples = samples[-FACE_MAX_SAMPLES:]
                        entry["samples"] = samples
                        entry["encoding"] = np.mean(samples, axis=0)
                        entry["last_seen"] = self.frame_count
                        entry["last_bbox_center"] = center

                    used_uids.add(match_id)
                    ids.append(match_id)

                    # Record stats
                    if center:
                        self.stats.record_sighting(match_id, center[0], center[1])

                    # Save face snapshot
                    if i < len(valid_locations):
                        self._maybe_save_snapshot(
                            rgb_image, valid_locations[i], match_id)

                except Exception as e:
                    print(f"Warning: face ID error: {e}")
                    ids.append("???")

            while len(ids) < len(valid_locations):
                ids.append("???")

            self._cleanup()
            self._maybe_merge()

        # Autosave outside lock
        if time.time() - self.last_save_time >= FACE_AUTOSAVE_INTERVAL:
            self.save_db()
            self.stats.save()

        return ids

    def _find_match(self, encoding, bbox_center, exclude_uids):
        """
        Find the closest matching known face using encoding distance
        weighted by spatial proximity.
        """
        if not self.known_faces:
            return None

        try:
            candidates = []
            for uid, entry in self.known_faces.items():
                if uid in exclude_uids:
                    continue
                stored = entry.get("encoding")
                if stored is None:
                    continue

                enc_dist = np.linalg.norm(stored - encoding)

                # Spatial bonus: if the face is near where we last saw this UID,
                # reduce the effective distance
                spatial_bonus = 0.0
                if bbox_center and entry.get("last_bbox_center"):
                    sx, sy = entry["last_bbox_center"]
                    cx, cy = bbox_center
                    spatial_dist = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
                    # spatial_dist is 0..~1.4 (normalized coords)
                    # Convert to a bonus: close = negative (helps match)
                    spatial_bonus = spatial_dist * FACE_SPATIAL_WEIGHT

                combined = enc_dist + spatial_bonus
                candidates.append((uid, combined, enc_dist))

            if not candidates:
                return None

            candidates.sort(key=lambda x: x[1])
            best_uid, best_combined, best_enc = candidates[0]

            if best_enc < self.match_threshold:
                return best_uid

            # Second pass: check individual samples for persisted faces
            for uid, entry in self.known_faces.items():
                if uid in exclude_uids or not entry.get("persisted"):
                    continue
                for sample in entry.get("samples", []):
                    try:
                        dist = np.linalg.norm(sample - encoding)
                        if dist < self.match_threshold and dist < best_enc:
                            best_enc = dist
                            best_uid = uid
                    except Exception:
                        continue

            return best_uid if best_enc < self.match_threshold else None

        except Exception as e:
            print(f"Warning: face matching error: {e}")
            return None

    def _maybe_merge(self):
        """
        Merge UIDs that have converged to the same person.
        This happens when someone gets two UIDs initially but
        the encodings become very similar over time.
        """
        try:
            uids = list(self.known_faces.keys())
            merge_map = {}  # old_uid -> keep_uid

            for i in range(len(uids)):
                for j in range(i + 1, len(uids)):
                    if uids[i] in merge_map or uids[j] in merge_map:
                        continue
                    ei = self.known_faces[uids[i]]["encoding"]
                    ej = self.known_faces[uids[j]]["encoding"]
                    dist = np.linalg.norm(ei - ej)

                    if dist < FACE_MERGE_THRESHOLD:
                        # Keep the one with more samples
                        si = len(self.known_faces[uids[i]].get("samples", []))
                        sj = len(self.known_faces[uids[j]].get("samples", []))
                        if si >= sj:
                            keep, discard = uids[i], uids[j]
                        else:
                            keep, discard = uids[j], uids[i]
                        merge_map[discard] = keep

            for discard, keep in merge_map.items():
                if discard in self.known_faces and keep in self.known_faces:
                    # Move samples
                    src = self.known_faces[discard]
                    dst = self.known_faces[keep]
                    combined = dst.get("samples", []) + src.get("samples", [])
                    if len(combined) > FACE_MAX_SAMPLES:
                        combined = combined[-FACE_MAX_SAMPLES:]
                    dst["samples"] = combined
                    dst["encoding"] = np.mean(combined, axis=0)
                    if src.get("persisted"):
                        dst["persisted"] = True
                    del self.known_faces[discard]
                    print(f"  Merged face {discard} -> {keep}")

        except Exception as e:
            print(f"Warning: merge error: {e}")

    # ----- Face Snapshots -----

    def _maybe_save_snapshot(self, rgb_image, face_loc, uid):
        """Save a face crop for this UID if enough time has passed."""
        try:
            if uid == "???":
                return

            now = time.time()
            last = self._snapshot_times.get(uid, 0)
            if now - last < FACE_SNAPSHOT_INTERVAL:
                return

            top, right, bottom, left = face_loc
            h, w = rgb_image.shape[:2]

            # Add padding
            face_h = bottom - top
            face_w = right - left
            if face_h < FACE_SNAPSHOT_MIN_SIZE or face_w < FACE_SNAPSHOT_MIN_SIZE:
                return

            pad_h = int(face_h * FACE_SNAPSHOT_PAD)
            pad_w = int(face_w * FACE_SNAPSHOT_PAD)
            top_p = max(0, top - pad_h)
            bottom_p = min(h, bottom + pad_h)
            left_p = max(0, left - pad_w)
            right_p = min(w, right + pad_w)

            face_crop = rgb_image[top_p:bottom_p, left_p:right_p]
            if face_crop.size == 0:
                return

            # Convert RGB to BGR for saving
            face_bgr = cv.cvtColor(face_crop, cv.COLOR_RGB2BGR)

            # Save to uid subfolder
            uid_dir = os.path.join(self._snapshot_dir, uid)
            os.makedirs(uid_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(uid_dir, filename)

            cv.imwrite(filepath, face_bgr, [cv.IMWRITE_JPEG_QUALITY, 90])
            self._snapshot_times[uid] = now

        except Exception as e:
            print(f"Warning: snapshot save error for {uid}: {e}")

    # ----- Cleanup -----

    def _cleanup(self):
        """Remove transient faces not seen recently. Persisted never expire."""
        try:
            expired = [
                uid for uid, entry in self.known_faces.items()
                if (self.frame_count - entry.get("last_seen", 0) > self.expiry_frames
                    and entry.get("last_seen", 0) > 0
                    and not entry.get("persisted", False))
            ]
            for uid in expired:
                del self.known_faces[uid]
        except Exception:
            pass

    # ----- Info -----

    def get_person_count(self):
        try:
            with self.lock:
                return sum(1 for e in self.known_faces.values()
                           if e.get("last_seen", 0) > 0)
        except Exception:
            return 0

    def get_total_known(self):
        with self.lock:
            return len(self.known_faces)
