#!/usr/bin/env python3
"""
Hand & Face Detection with Persistent Identity Tracking.

Supports one or two cameras. Camera 1 uses RGB processing, Camera 2
uses IR-optimized processing. Both share a single face identity database
so the same person gets the same UUID on either camera.

Usage:
    python main.py 0                    # single RGB webcam
    python main.py 0 1                  # cam 0 (RGB) + cam 1 (IR)
    python main.py 0 --mode1 ir         # single IR camera
    python main.py 0 1 --mode2 rgb      # both RGB

Press 'q' or ESC to quit.
"""

# Suppress noisy MediaPipe/absl warnings before any imports
import os
os.environ["GLOG_minloglevel"] = "2"          # suppress INFO and WARNING from glog
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # suppress TF warnings if present
os.environ["GRPC_VERBOSITY"] = "ERROR"

import argparse
import sys
import time
import traceback

import numpy as np
import cv2 as cv

from config import TARGET_FPS, FRAME_DELAY, FACE_DB_PATH
from gpu_utils import check_gpu, print_gpu_status
from face_tracker import FaceTracker
from pipeline import CameraPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hand & Face Detection with Persistent Identity Tracking")

    parser.add_argument("source1",
                        help="First camera index or video file path")
    parser.add_argument("source2", nargs="?", default=None,
                        help="Second camera index or video file path")
    parser.add_argument("--mode1", default="rgb", choices=["rgb", "ir"],
                        help="Processing mode for camera 1 (default: rgb)")
    parser.add_argument("--mode2", default="ir", choices=["rgb", "ir"],
                        help="Processing mode for camera 2 (default: ir)")
    parser.add_argument("--db", default=FACE_DB_PATH,
                        help="Face database file path (shared across cameras)")
    parser.add_argument("--fps", type=int, default=TARGET_FPS,
                        help=f"Target FPS (default: {TARGET_FPS})")

    args = parser.parse_args()

    # Convert numeric strings to ints
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


def resize_to_match(img, target_h, target_w):
    """Resize an image to target dimensions."""
    try:
        if img is None or img.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return cv.resize(img, (target_w, target_h))
    except Exception:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)


def build_grid(frames, target_h=480, target_w=640):
    """Stack camera frames into a vertical grid."""
    rows = []
    for orig, proc in frames:
        orig_r = resize_to_match(orig, target_h, target_w)
        proc_r = resize_to_match(proc, target_h, target_w)
        # Ensure both are BGR 3-channel
        if len(orig_r.shape) == 2:
            orig_r = cv.cvtColor(orig_r, cv.COLOR_GRAY2BGR)
        if len(proc_r.shape) == 2:
            proc_r = cv.cvtColor(proc_r, cv.COLOR_GRAY2BGR)
        rows.append(np.hstack([orig_r, proc_r]))
    return np.vstack(rows) if rows else np.zeros((target_h, target_w * 2, 3), dtype=np.uint8)


def main():
    args = parse_args()

    # --- GPU detection ---
    gpu_info = check_gpu()
    print_gpu_status(gpu_info)

    use_opencv_cuda = gpu_info["opencv_cuda"]
    use_dlib_cuda = gpu_info["dlib_cuda"]

    # --- Shared face tracker ---
    print(f"\nFace database: {args.db}")
    tracker = FaceTracker(use_cuda=use_dlib_cuda, db_path=args.db)
    print(f"  Total known identities: {tracker.get_total_known()}")

    # --- Build camera list ---
    cameras = [(args.source1, args.mode1, "Cam1")]
    if args.source2 is not None:
        cameras.append((args.source2, args.mode2, "Cam2"))

    # --- Initialize pipelines ---
    frame_delay = 1.0 / args.fps
    num_cams = len(cameras)
    print(f"\nInitializing {num_cams} camera(s)...")

    pipelines = []
    for source, mode, label in cameras:
        full_label = f"{label} [{mode.upper()}] (src: {source})"
        try:
            pipeline = CameraPipeline(
                source=source,
                tracker=tracker,
                label=full_label,
                mode=mode,
                use_opencv_cuda=use_opencv_cuda,
            )
            pipelines.append(pipeline)
            print(f"  [{full_label}] Ready")
        except RuntimeError as e:
            print(f"  ERROR opening {full_label}: {e}")
            for p in pipelines:
                p.close()
            sys.exit(1)

    # --- Window title with GPU info ---
    gpu_name = gpu_info.get("gpu_name", "None")
    if gpu_name == "None":
        title = f"Hand & Face Detection — CPU only"
    else:
        mem = gpu_info.get("gpu_memory", "?")
        title = f"Hand & Face Detection — {gpu_name} ({mem})"
    if num_cams > 1:
        title += f" — {num_cams} cameras"

    cv.namedWindow(title, cv.WINDOW_NORMAL)

    print(f"\nRunning {num_cams} camera(s) at ~{args.fps} FPS...")
    print(f"Window: {title}")
    print("Press 'q' or ESC to quit.\n")

    # Frame timing diagnostics
    slow_frame_count = 0
    frame_number = 0

    try:
        while True:
            loop_start = time.time()
            frames = []
            all_dead = True
            frame_number += 1

            for pipeline in pipelines:
                if not pipeline.is_healthy:
                    err_frame = pipeline._make_error_frame(
                        msg=f"{pipeline.label}: camera lost")
                    frames.append((err_frame, err_frame.copy()))
                    continue

                all_dead = False

                t0 = time.time()
                ret, img = pipeline.read_frame()
                read_ms = (time.time() - t0) * 1000

                if not ret or img is None:
                    err_frame = pipeline._make_error_frame(
                        msg=f"{pipeline.label}: no frame")
                    frames.append((err_frame, err_frame.copy()))
                    continue

                t0 = time.time()
                try:
                    orig, proc = pipeline.process(img)
                    frames.append((orig, proc))
                except Exception as e:
                    print(f"  [{pipeline.label}] process() crash: {e}")
                    traceback.print_exc()
                    err_frame = pipeline._make_error_frame(
                        msg=f"{pipeline.label}: error")
                    frames.append((err_frame, err_frame.copy()))
                process_ms = (time.time() - t0) * 1000

                # Warn about slow frames (but not too often)
                total_ms = read_ms + process_ms
                if total_ms > frame_delay * 1000 * 2:
                    slow_frame_count += 1
                    if slow_frame_count <= 5 or slow_frame_count % 50 == 0:
                        print(f"  [{pipeline.label}] Slow frame #{frame_number}: "
                              f"read={read_ms:.0f}ms process={process_ms:.0f}ms "
                              f"total={total_ms:.0f}ms")

            if all_dead:
                print("All cameras lost. Exiting.")
                break

            if not frames:
                # Still call waitKey to keep window responsive
                cv.waitKey(1)
                continue

            try:
                ref_h, ref_w = frames[0][0].shape[:2]
                display = build_grid(frames, target_h=ref_h, target_w=ref_w)
                cv.imshow(title, display)
            except Exception as e:
                print(f"  Display error: {e}")

            # Always call waitKey with at least 1ms to keep X11 alive
            elapsed = time.time() - loop_start
            wait_ms = max(1, int((frame_delay - elapsed) * 1000))
            key = cv.waitKey(wait_ms) & 0xFF
            if key in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()

    finally:
        print("\nShutting down...")
        try:
            tracker.save_db()
            tracker.stats.save()
        except Exception as e:
            print(f"  Warning: could not save data: {e}")

        for pipeline in pipelines:
            pipeline.close()

        cv.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
