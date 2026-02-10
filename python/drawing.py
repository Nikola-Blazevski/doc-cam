"""Drawing overlays for hands, faces, and status HUD."""

import numpy as np
import cv2 as cv

from config import (
    HAND_CONNECTIONS, FINGERS, FINGER_CONNECTIONS,
    FINGER_LANDMARKS, FACE_KEYPOINT_NAMES,
    POSE_CONNECTIONS, POSE_COLOR,
)


def color_for_uuid(uid):
    """Generate a consistent BGR color from a UUID string."""
    try:
        h = hash(uid) % 360
        color_hsv = np.uint8([[[h // 2, 200, 255]]])
        color_bgr = cv.cvtColor(color_hsv, cv.COLOR_HSV2BGR)[0][0]
        return int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
    except Exception:
        return (0, 255, 0)


def _ensure_bgr(image):
    """Convert grayscale to BGR if needed, otherwise copy."""
    try:
        if image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        if len(image.shape) == 2:
            return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        if image.shape[2] == 1:
            return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        return np.copy(image)
    except Exception:
        return np.zeros((480, 640, 3), dtype=np.uint8)


def _clamp(val, lo, hi):
    """Clamp integer to range."""
    return max(lo, min(int(val), hi))


def draw_hand_overlay(image, hand_result):
    """Draw hand skeleton, finger labels, and bounding boxes."""
    if hand_result is None:
        return _ensure_bgr(image)

    h, w = image.shape[:2]
    annotated = _ensure_bgr(image)

    try:
        for idx in range(len(hand_result.hand_landmarks)):
            hand_landmarks = hand_result.hand_landmarks[idx]
            handedness = hand_result.handedness[idx]

            # Full skeleton in gray
            for start, end in HAND_CONNECTIONS:
                try:
                    x1 = _clamp(hand_landmarks[start].x * w, 0, w - 1)
                    y1 = _clamp(hand_landmarks[start].y * h, 0, h - 1)
                    x2 = _clamp(hand_landmarks[end].x * w, 0, w - 1)
                    y2 = _clamp(hand_landmarks[end].y * h, 0, h - 1)
                    cv.line(annotated, (x1, y1), (x2, y2), (180, 180, 180), 1)
                except (IndexError, TypeError):
                    continue

            for lm in hand_landmarks:
                try:
                    cx = _clamp(lm.x * w, 0, w - 1)
                    cy = _clamp(lm.y * h, 0, h - 1)
                    cv.circle(annotated, (cx, cy), 3, (150, 150, 150), -1)
                except (TypeError, ValueError):
                    continue

            # Highlighted fingers
            for name, tip_idx, color in FINGERS:
                try:
                    for start, end in FINGER_CONNECTIONS[name]:
                        x1 = _clamp(hand_landmarks[start].x * w, 0, w - 1)
                        y1 = _clamp(hand_landmarks[start].y * h, 0, h - 1)
                        x2 = _clamp(hand_landmarks[end].x * w, 0, w - 1)
                        y2 = _clamp(hand_landmarks[end].y * h, 0, h - 1)
                        cv.line(annotated, (x1, y1), (x2, y2), color, 3)

                    for li in FINGER_LANDMARKS[name]:
                        cx = _clamp(hand_landmarks[li].x * w, 0, w - 1)
                        cy = _clamp(hand_landmarks[li].y * h, 0, h - 1)
                        cv.circle(annotated, (cx, cy), 5, color, -1)

                    tip = hand_landmarks[tip_idx]
                    tx = _clamp(tip.x * w, 0, w - 1)
                    ty = _clamp(tip.y * h, 0, h - 1)
                    (tw_t, th_t), _ = cv.getTextSize(name, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv.rectangle(annotated,
                                 (_clamp(tx - 2, 0, w), _clamp(ty - th_t - 8, 0, h)),
                                 (_clamp(tx + tw_t + 2, 0, w), _clamp(ty - 2, 0, h)),
                                 (0, 0, 0), -1)
                    cv.putText(annotated, name, (tx, _clamp(ty - 5, 0, h)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except (IndexError, TypeError, ValueError):
                    continue

            # Bounding box
            try:
                xs = [lm.x for lm in hand_landmarks]
                ys = [lm.y for lm in hand_landmarks]
                x_min = _clamp(min(xs) * w - 20, 0, w - 1)
                y_min = _clamp(min(ys) * h - 20, 0, h - 1)
                x_max = _clamp(max(xs) * w + 20, 0, w - 1)
                y_max = _clamp(max(ys) * h + 20, 0, h - 1)
                cv.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                label = handedness[0]
                cv.putText(annotated, f"{label.category_name} ({label.score:.0%})",
                           (x_min, _clamp(y_min - 10, 10, h)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (88, 205, 54), 2)
            except Exception:
                pass

    except Exception as e:
        # Don't crash — just return what we have
        pass

    return annotated


def draw_face_overlay(image, face_result, face_ids, img_h, img_w):
    """Draw face bounding boxes, UUIDs, confidence, and keypoints."""
    if face_result is None:
        return _ensure_bgr(image)

    annotated = _ensure_bgr(image)

    try:
        for i, detection in enumerate(face_result.detections):
            try:
                bbox = detection.bounding_box
                x = _clamp(bbox.origin_x, 0, img_w - 1)
                y = _clamp(bbox.origin_y, 0, img_h - 1)
                bw = _clamp(bbox.width, 1, img_w - x)
                bh = _clamp(bbox.height, 1, img_h - y)

                uid = face_ids[i] if i < len(face_ids) else "???"
                color = color_for_uuid(uid) if uid != "???" else (255, 0, 255)

                # Bounding box
                cv.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

                # Labels
                score = detection.categories[0].score if detection.categories else 0
                id_text = f"ID: {uid}"
                conf_text = f"Face ({score:.0%})"

                (tw1, th1), _ = cv.getTextSize(id_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                (tw2, th2), _ = cv.getTextSize(conf_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                bg_top = _clamp(y - th1 - th2 - 20, 0, img_h)
                bg_right = _clamp(x + max(tw1, tw2) + 8, 0, img_w)
                cv.rectangle(annotated, (x, bg_top), (bg_right, y), (0, 0, 0), -1)
                cv.putText(annotated, id_text,
                           (x + 4, _clamp(y - th2 - 12, 10, img_h)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv.putText(annotated, conf_text,
                           (x + 4, _clamp(y - 4, 10, img_h)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Keypoints
                if detection.keypoints:
                    for j, kp in enumerate(detection.keypoints):
                        cx = _clamp(kp.x * img_w, 0, img_w - 1)
                        cy = _clamp(kp.y * img_h, 0, img_h - 1)
                        cv.circle(annotated, (cx, cy), 4, color, -1)

            except Exception:
                continue

    except Exception:
        pass

    return annotated


def draw_hud(image, fps, person_count, gpu_status, label=None, body_count=0):
    """Draw status overlay (FPS, people count, GPU/mode status)."""
    try:
        annotated = image if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        lines = []
        if label:
            lines.append(label)
        lines += [
            f"FPS: {fps}",
            f"Faces: {person_count}  Bodies: {body_count}",
            f"CV: {'CUDA' if gpu_status.get('opencv_cuda') else 'CPU'}",
            f"Mode: {gpu_status.get('mode', 'rgb').upper()}",
        ]
        for i, line in enumerate(lines):
            y_pos = 25 + i * 25
            cv.putText(annotated, line, (10, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv.putText(annotated, line, (10, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return annotated
    except Exception:
        return image


def draw_pose_overlay(image, pose_result):
    """Draw body pose skeleton overlays."""
    if pose_result is None:
        return _ensure_bgr(image)

    h, w = image.shape[:2]
    annotated = _ensure_bgr(image)

    try:
        for pose_idx in range(len(pose_result.pose_landmarks)):
            landmarks = pose_result.pose_landmarks[pose_idx]

            # Draw connections
            for start, end in POSE_CONNECTIONS:
                try:
                    if start >= len(landmarks) or end >= len(landmarks):
                        continue
                    lm1 = landmarks[start]
                    lm2 = landmarks[end]
                    # Skip low-visibility landmarks
                    if lm1.visibility < 0.5 or lm2.visibility < 0.5:
                        continue
                    x1 = _clamp(lm1.x * w, 0, w - 1)
                    y1 = _clamp(lm1.y * h, 0, h - 1)
                    x2 = _clamp(lm2.x * w, 0, w - 1)
                    y2 = _clamp(lm2.y * h, 0, h - 1)
                    cv.line(annotated, (x1, y1), (x2, y2), POSE_COLOR, 2)
                except (IndexError, TypeError):
                    continue

            # Draw landmark dots
            for i, lm in enumerate(landmarks):
                try:
                    if lm.visibility < 0.5:
                        continue
                    cx = _clamp(lm.x * w, 0, w - 1)
                    cy = _clamp(lm.y * h, 0, h - 1)
                    # Larger dots for key joints
                    radius = 5 if i in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28) else 3
                    cv.circle(annotated, (cx, cy), radius, POSE_COLOR, -1)
                except (TypeError, ValueError):
                    continue

            # Body bounding box
            try:
                visible = [(lm.x, lm.y) for lm in landmarks if lm.visibility >= 0.5]
                if len(visible) >= 4:
                    xs = [p[0] for p in visible]
                    ys = [p[1] for p in visible]
                    x_min = _clamp(min(xs) * w - 15, 0, w - 1)
                    y_min = _clamp(min(ys) * h - 15, 0, h - 1)
                    x_max = _clamp(max(xs) * w + 15, 0, w - 1)
                    y_max = _clamp(max(ys) * h + 15, 0, h - 1)
                    cv.rectangle(annotated, (x_min, y_min), (x_max, y_max),
                                 POSE_COLOR, 1)
                    cv.putText(annotated, f"Body {pose_idx + 1}",
                               (x_min, _clamp(y_min - 8, 10, h)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, POSE_COLOR, 2)
            except Exception:
                pass

    except Exception:
        pass

    return annotated
