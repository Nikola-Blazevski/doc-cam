#!/bin/bash
# Download required model files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading hand landmarker model..."
wget -q --show-progress -O hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

echo "Downloading face detector model..."
wget -q --show-progress -O blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

echo "Downloading pose landmarker model..."
wget -q --show-progress -O pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

echo ""
echo "Done. Models downloaded:"
ls -lh hand_landmarker.task blaze_face_short_range.tflite pose_landmarker_lite.task
