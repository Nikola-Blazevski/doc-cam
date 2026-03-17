"""
Stats collector for face tracking.

Records sighting events per UUID:
  - timestamp (ISO format)
  - bbox center position (normalized 0..1)
  - hour of day

Persists to a JSON file. The UI reads this for analytics.
"""

import json
import os
import time
import threading
from datetime import datetime
from collections import defaultdict

from config import STATS_DB_PATH, STATS_RECORD_INTERVAL


class StatsCollector:
    """Thread-safe stats collection for face sightings."""

    def __init__(self, db_path=STATS_DB_PATH):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._last_record_time = {}  # uid -> last record timestamp

        # In-memory stats: loaded from disk + new data
        # Format: { uid: { "sightings": [...], "positions": [...] } }
        self.data = self._load()

    def _load(self):
        """Load existing stats from disk."""
        if not os.path.exists(self.db_path):
            return {}
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"Warning: could not load stats: {e}")
        return {}

    def save(self):
        """Persist stats to disk."""
        with self.lock:
            try:
                tmp = self.db_path + ".tmp"
                with open(tmp, 'w') as f:
                    json.dump(self.data, f, indent=None, separators=(',', ':'))
                os.replace(tmp, self.db_path)
            except Exception as e:
                print(f"Warning: could not save stats: {e}")

    def record_sighting(self, uid, bbox_center_x, bbox_center_y):
        """
        Record a face sighting event.

        Args:
            uid: person UUID
            bbox_center_x: normalized x position (0..1, left to right)
            bbox_center_y: normalized y position (0..1, top to bottom)
        """
        if uid == "???":
            return

        now = time.time()
        with self.lock:
            # Rate limit per person
            last = self._last_record_time.get(uid, 0)
            if now - last < STATS_RECORD_INTERVAL:
                return
            self._last_record_time[uid] = now

            if uid not in self.data:
                self.data[uid] = {"sightings": [], "positions": []}

            entry = self.data[uid]

            dt = datetime.now()
            entry["sightings"].append({
                "ts": dt.isoformat(timespec='seconds'),
                "hour": dt.hour,
                "weekday": dt.weekday(),  # 0=Mon, 6=Sun
                "day": dt.strftime("%Y-%m-%d"),
            })

            entry["positions"].append({
                "x": round(bbox_center_x, 3),
                "y": round(bbox_center_y, 3),
            })

            # Cap stored data per person (keep last 5000 entries)
            if len(entry["sightings"]) > 5000:
                entry["sightings"] = entry["sightings"][-5000:]
            if len(entry["positions"]) > 5000:
                entry["positions"] = entry["positions"][-5000:]

    def get_stats(self):
        """Return full stats data for the API."""
        with self.lock:
            return dict(self.data)

    def get_person_stats(self, uid):
        """Return stats for a single person."""
        with self.lock:
            return self.data.get(uid, {"sightings": [], "positions": []})
