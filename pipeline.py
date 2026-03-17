"""
Single camera processing pipeline — detection, tracking, and drawing.

Face recognition runs in a persistent background worker thread.
Pose (body) detection runs inline since it's fast on GPU.
"""

import time
import threading

import numpy as np
import cv2 as cv
import mediapipe as mp

from config import FRAME_DELAY
from processing import get_processor
from drawing import draw_hand_overlay, draw_pose_overlay, draw_hud
from detectors import create_hand_detector, create_face_detector, create_pose_detector


class CameraPipeline:
    """
    Encapsulates one camera source with its own detectors.
    Face recognition runs async. Hands, face detection, and pose run inline.
    """

    def __init__(self, source, tracker, label="Camera",
                 mode="rgb", use_opencv_cuda=False):
        self.label = label
        self.source = source
        self.mode = mode
        self.tracker = tracker
        self.frame_timestamp_ms = 0
        self.gpu_info = {
            "opencv_cuda": use_opencv_cuda,
            "mode": mode,
        }
        self._consecutive_errors = 0
        self._max_consecutive_errors = 30

        # Open capture
        self.cap = cv.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        # Detectors
        print(f"\n  [{label}] Initializing detectors (mode: {mode})...")
        self.hand_detector = create_hand_detector(use_gpu=True)
        self.face_detector = create_face_detector(use_gpu=True)

        # Pose detector (may fail if model not downloaded)
        self.pose_detector = None
        try:
            self.pose_detector = create_pose_detector(use_gpu=True)
        except Exception as e:
            print(f"  [{label}] Pose detector unavailable: {e}")
            print(f"  [{label}] Run download_models.sh to get the pose model")

        self.process_frame = get_processor(mode=mode, use_cuda=use_opencv_cuda)

        # --- Async face recognition ---
        self._face_lock = threading.Lock()
        self._pending_work = None
        self._result_face_ids = []
        self._result_face_bboxes = []
        self._result_face_scores = []
        self._worker_stop = threading.Event()
        self._worker_wake = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._face_worker_loop, daemon=True)
        self._worker_thread.start()

        # Pose result for HUD
        self._last_body_count = 0

        # FPS tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.display_fps = 0

    # ----------------------------------------------------------------
    # Background face recognition worker
    # ----------------------------------------------------------------

    def _face_worker_loop(self):
        while not self._worker_stop.is_set():
            self._worker_wake.wait(timeout=0.5)
            self._worker_wake.clear()

            if self._worker_stop.is_set():
                break

            with self._face_lock:
                work = self._pending_work
                self._pending_work = None

            if work is None:
                continue

            rgb, face_locations, bboxes, scores = work

            try:
                face_ids = self.tracker.identify(rgb, face_locations)
            except Exception as e:
                print(f"  [{self.label}] BG face recognition error: {e}")
                face_ids = ["???"] * len(face_locations)

            with self._face_lock:
                self._result_face_ids = face_ids
                self._result_face_bboxes = bboxes
                self._result_face_scores = scores

    def _submit_face_work(self, rgb, face_locations, bboxes, scores):
        with self._face_lock:
            self._pending_work = (rgb.copy(), list(face_locations),
                                  list(bboxes), list(scores))
        self._worker_wake.set()

    def _get_face_results(self):
        with self._face_lock:
            return (list(self._result_face_ids),
                    list(self._result_face_bboxes),
                    list(self._result_face_scores))

    # ----------------------------------------------------------------
    # Frame I/O
    # ----------------------------------------------------------------

    def read_frame(self):
        try:
            ret, img = self.cap.read()
            if not ret:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                self.frame_timestamp_ms = 0
                ret, img = self.cap.read()
            if ret and img is not None and img.size > 0:
                self._consecutive_errors = 0
                return True, img
            else:
                self._consecutive_errors += 1
                return False, None
        except Exception as e:
            print(f"  [{self.label}] Frame read error: {e}")
            self._consecutive_errors += 1
            return False, None

    def _ensure_bgr(self, img):
        try:
            if img is None or img.size == 0:
                return None
            if len(img.shape) == 2:
                return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if img.shape[2] == 1:
                return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                return cv.cvtColor(img, cv.COLOR_BGRA2BGR)
            return img
        except Exception:
            return None

    def _make_error_frame(self, h=480, w=640, msg="Error"):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv.putText(frame, msg, (20, h // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    # ----------------------------------------------------------------
    # Main processing
    # ----------------------------------------------------------------

    def process(self, img):
        img_bgr = self._ensure_bgr(img)
        if img_bgr is None:
            err = self._make_error_frame(msg=f"{self.label}: bad frame")
            return err, err.copy()

        h, w = img_bgr.shape[:2]

        # --- Stage 1: Image processing ---
        try:
            processed = self.process_frame(img_bgr)
        except Exception as e:
            print(f"  [{self.label}] Processing error: {e}")
            processed = cv.cvtColor(
                cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

        # --- Stage 2: Prepare MediaPipe image ---
        rgb = None
        mp_image = None
        try:
            rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            if not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.frame_timestamp_ms += int(FRAME_DELAY * 1000)
        except Exception as e:
            print(f"  [{self.label}] MediaPipe image error: {e}")

        # --- Stage 3: Hand detection ---
        hand_result = None
        try:
            if mp_image is not None:
                hand_result = self.hand_detector.detect_for_video(
                    mp_image, self.frame_timestamp_ms)
        except Exception as e:
            print(f"  [{self.label}] Hand detection error: {e}")

        # --- Stage 4: Pose detection ---
        pose_result = None
        try:
            if self.pose_detector and mp_image is not None:
                pose_result = self.pose_detector.detect_for_video(
                    mp_image, self.frame_timestamp_ms)
                self._last_body_count = (
                    len(pose_result.pose_landmarks) if pose_result else 0)
        except Exception as e:
            print(f"  [{self.label}] Pose detection error: {e}")

        # --- Stage 5: Face detection + submit async recognition ---
        face_locations = []
        current_bboxes = []
        current_scores = []

        try:
            if mp_image is not None:
                face_result = self.face_detector.detect_for_video(
                    mp_image, self.frame_timestamp_ms)

                if face_result and face_result.detections:
                    for det in face_result.detections:
                        bbox = det.bounding_box
                        top = max(0, bbox.origin_y)
                        right = min(w, bbox.origin_x + bbox.width)
                        bottom = min(h, bbox.origin_y + bbox.height)
                        left = max(0, bbox.origin_x)
                        if bottom - top >= 10 and right - left >= 10:
                            face_locations.append((top, right, bottom, left))
                            current_bboxes.append((left, top,
                                                   right - left, bottom - top))
                            score = (det.categories[0].score
                                     if det.categories else 0.0)
                            current_scores.append(score)
        except Exception as e:
            print(f"  [{self.label}] Face detection error: {e}")

        if face_locations and rgb is not None:
            try:
                self._submit_face_work(rgb, face_locations,
                                       current_bboxes, current_scores)
            except Exception as e:
                print(f"  [{self.label}] Face submit error: {e}")

        # --- Stage 6: Draw ---
        face_ids, draw_bboxes, draw_scores = self._get_face_results()

        original_annotated = img_bgr.copy()

        try:
            if pose_result:
                original_annotated = draw_pose_overlay(
                    original_annotated, pose_result)
        except Exception as e:
            print(f"  [{self.label}] Pose drawing error: {e}")

        try:
            if hand_result:
                original_annotated = draw_hand_overlay(
                    original_annotated, hand_result)
        except Exception as e:
            print(f"  [{self.label}] Hand drawing error: {e}")

        try:
            if draw_bboxes and face_ids:
                original_annotated = _draw_face_boxes(
                    original_annotated, face_ids, draw_bboxes, draw_scores)
        except Exception as e:
            print(f"  [{self.label}] Face drawing error: {e}")

        try:
            original_annotated = draw_hud(
                original_annotated, self.display_fps,
                self.tracker.get_person_count(), self.gpu_info, self.label,
                body_count=self._last_body_count)
        except Exception as e:
            print(f"  [{self.label}] HUD drawing error: {e}")

        # Processed view
        try:
            processed_annotated = processed
            if pose_result:
                processed_annotated = draw_pose_overlay(
                    processed_annotated, pose_result)
            if hand_result:
                processed_annotated = draw_hand_overlay(
                    processed_annotated, hand_result)
            if draw_bboxes and face_ids:
                processed_annotated = _draw_face_boxes(
                    processed_annotated, face_ids, draw_bboxes, draw_scores)
        except Exception:
            processed_annotated = processed

        # Ensure BGR
        try:
            if processed_annotated is not None:
                if len(processed_annotated.shape) == 2:
                    processed_annotated = cv.cvtColor(
                        processed_annotated, cv.COLOR_GRAY2BGR)
            else:
                processed_annotated = original_annotated.copy()
        except Exception:
            processed_annotated = original_annotated.copy()

        # FPS
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_timer >= 1.0:
            self.display_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = now

        return original_annotated, processed_annotated

    @property
    def is_healthy(self):
        return self._consecutive_errors < self._max_consecutive_errors

    def close(self):
        self._worker_stop.set()
        self._worker_wake.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

        for det in (self.hand_detector, self.face_detector, self.pose_detector):
            try:
                if det:
                    det.close()
            except Exception:
                pass
        try:
            self.cap.release()
        except Exception:
            pass


# -----------------------------------------------------------------
# Standalone face drawing from raw bbox data
# -----------------------------------------------------------------

def _draw_face_boxes(image, face_ids, bboxes, scores):
    from drawing import color_for_uuid, _ensure_bgr, _clamp

    annotated = _ensure_bgr(image)
    h, w = annotated.shape[:2]

    for i, (x, y, bw, bh) in enumerate(bboxes):
        try:
            uid = face_ids[i] if i < len(face_ids) else "???"
            score = scores[i] if i < len(scores) else 0.0
            color = color_for_uuid(uid) if uid != "???" else (255, 0, 255)

            x = _clamp(x, 0, w - 1)
            y = _clamp(y, 0, h - 1)
            bw = _clamp(bw, 1, w - x)
            bh = _clamp(bh, 1, h - y)

            cv.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

            id_text = f"ID: {uid}"
            conf_text = f"Face ({score:.0%})"

            (tw1, th1), _ = cv.getTextSize(
                id_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (tw2, th2), _ = cv.getTextSize(
                conf_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            bg_top = _clamp(y - th1 - th2 - 20, 0, h)
            bg_right = _clamp(x + max(tw1, tw2) + 8, 0, w)
            cv.rectangle(annotated, (x, bg_top), (bg_right, y), (0, 0, 0), -1)
            cv.putText(annotated, id_text,
                       (x + 4, _clamp(y - th2 - 12, 10, h)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv.putText(annotated, conf_text,
                       (x + 4, _clamp(y - 4, 10, h)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        except Exception:
            continue

    return annotated
