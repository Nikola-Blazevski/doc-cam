"""MediaPipe detector initialization with automatic GPU-to-CPU fallback."""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import HAND_MODEL_PATH, FACE_MODEL_PATH, POSE_MODEL_PATH


def create_hand_detector(use_gpu=True):
    """Create a HandLandmarker, attempting GPU first then falling back to CPU."""
    delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
    try:
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=HAND_MODEL_PATH,
                delegate=delegate,
            ),
            num_hands=4,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        mode = "GPU" if use_gpu else "CPU"
        print(f"  Hand detector: {mode}")
        return detector
    except Exception as e:
        if use_gpu:
            print(f"  Hand detector GPU failed ({e}), falling back to CPU...")
            return create_hand_detector(use_gpu=False)
        raise


def create_face_detector(use_gpu=True):
    """Create a FaceDetector, attempting GPU first then falling back to CPU."""
    delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
    try:
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(
                model_asset_path=FACE_MODEL_PATH,
                delegate=delegate,
            ),
            min_detection_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.FaceDetector.create_from_options(options)
        mode = "GPU" if use_gpu else "CPU"
        print(f"  Face detector: {mode}")
        return detector
    except Exception as e:
        if use_gpu:
            print(f"  Face detector GPU failed ({e}), falling back to CPU...")
            return create_face_detector(use_gpu=False)
        raise


def create_pose_detector(use_gpu=True):
    """Create a PoseLandmarker, attempting GPU first then falling back to CPU."""
    delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
    try:
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=POSE_MODEL_PATH,
                delegate=delegate,
            ),
            num_poses=4,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        mode = "GPU" if use_gpu else "CPU"
        print(f"  Pose detector: {mode}")
        return detector
    except Exception as e:
        if use_gpu:
            print(f"  Pose detector GPU failed ({e}), falling back to CPU...")
            return create_pose_detector(use_gpu=False)
        raise
