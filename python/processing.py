"""Image processing pipelines for RGB and IR cameras, with CPU/GPU paths."""

import numpy as np
import cv2 as cv

from config import (
    SKIN_LOWER, SKIN_UPPER,
    IR_CLAHE_CLIP, IR_CLAHE_GRID, IR_BLUR_KSIZE,
    IR_THRESH_BLOCK, IR_THRESH_C, IR_MORPH_KSIZE,
    IR_MIN_CONTOUR_AREA,
)


# ===================================================================
# RGB processing (skin-color segmentation)
# ===================================================================

def process_rgb_cpu(img):
    """Skin-color segmentation + adaptive threshold on CPU."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:4]
    mask = np.zeros(mask.shape, dtype=np.uint8)
    for c in contours:
        cv.drawContours(mask, [c], 0, 255, -1)

    mask = cv.GaussianBlur(mask, (5, 5), 0)
    hand = cv.bitwise_and(img, img, mask=mask)
    hand_gray = cv.cvtColor(hand, cv.COLOR_BGR2GRAY)
    hand_thresh = cv.adaptiveThreshold(
        hand_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return hand_thresh


def process_rgb_gpu(img):
    """Skin-color segmentation + adaptive threshold with CUDA acceleration."""
    gpu_img = cv.cuda_GpuMat()
    gpu_img.upload(img)

    gpu_hsv = cv.cuda.cvtColor(gpu_img, cv.COLOR_BGR2HSV)
    hsv = gpu_hsv.download()

    mask = cv.inRange(hsv, SKIN_LOWER, SKIN_UPPER)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    morph_filter = cv.cuda.createMorphologyFilter(cv.MORPH_CLOSE, cv.CV_8UC1, kernel)
    gpu_mask = cv.cuda_GpuMat()
    gpu_mask.upload(mask)
    gpu_mask = morph_filter.apply(gpu_mask)
    mask = gpu_mask.download()

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:4]
    mask = np.zeros(mask.shape, dtype=np.uint8)
    for c in contours:
        cv.drawContours(mask, [c], 0, 255, -1)

    gpu_mask.upload(mask)
    gauss_filter = cv.cuda.createGaussianFilter(cv.CV_8UC1, cv.CV_8UC1, (5, 5), 0)
    gpu_mask = gauss_filter.apply(gpu_mask)
    mask = gpu_mask.download()

    hand = cv.bitwise_and(img, img, mask=mask)
    hand_gray = cv.cvtColor(hand, cv.COLOR_BGR2GRAY)
    hand_thresh = cv.adaptiveThreshold(
        hand_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return hand_thresh


# ===================================================================
# IR processing (intensity + edge based)
# ===================================================================
#
# IR cameras produce grayscale-like images where:
#   - Skin/body appears BRIGHT (warm objects emit more IR)
#   - Background appears DARK (cooler objects)
# HSV skin-color segmentation fails because there is no color info.
#
# Strategy:
#   1. Convert to true grayscale
#   2. CLAHE to normalize uneven IR illumination
#   3. Gaussian blur to reduce sensor noise
#   4. Intensity threshold to isolate warm regions (skin/body)
#   5. Morphological cleanup
#   6. Contour filtering by area
#   7. Adaptive threshold on the segmented region for detail
# ===================================================================

def process_ir_cpu(img):
    """IR-optimized processing on CPU."""
    # Handle both grayscale and BGR input from IR cameras
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 1. CLAHE — normalize uneven IR illumination
    clahe = cv.createCLAHE(clipLimit=IR_CLAHE_CLIP, tileGridSize=IR_CLAHE_GRID)
    enhanced = clahe.apply(gray)

    # 2. Blur to reduce IR sensor noise
    blurred = cv.GaussianBlur(enhanced, IR_BLUR_KSIZE, 0)

    # 3. Otsu threshold — automatically finds the best split between
    #    warm (bright skin) and cool (dark background) regions
    _, mask = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 4. Morphological cleanup — close small gaps, remove noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                      (IR_MORPH_KSIZE, IR_MORPH_KSIZE))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    # 5. Contour filtering — keep only large warm blobs
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) > IR_MIN_CONTOUR_AREA]
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:6]

    mask = np.zeros(mask.shape, dtype=np.uint8)
    for c in contours:
        cv.drawContours(mask, [c], 0, 255, -1)

    mask = cv.GaussianBlur(mask, (5, 5), 0)

    # 6. Segment and adaptive threshold for detail
    segmented = cv.bitwise_and(enhanced, enhanced, mask=mask)
    result = cv.adaptiveThreshold(
        segmented, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, IR_THRESH_BLOCK, IR_THRESH_C)

    return result


def process_ir_gpu(img):
    """IR-optimized processing with CUDA acceleration where possible."""
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # CLAHE on GPU
    gpu_gray = cv.cuda_GpuMat()
    gpu_gray.upload(gray)
    clahe = cv.cuda.createCLAHE(clipLimit=IR_CLAHE_CLIP, tileGridSize=IR_CLAHE_GRID)
    gpu_enhanced = clahe.apply(gpu_gray, cv.cuda.Stream.Null())
    enhanced = gpu_enhanced.download()

    # Gaussian blur on GPU
    gpu_enhanced2 = cv.cuda_GpuMat()
    gpu_enhanced2.upload(enhanced)
    gauss = cv.cuda.createGaussianFilter(cv.CV_8UC1, cv.CV_8UC1, IR_BLUR_KSIZE, 0)
    gpu_blurred = gauss.apply(gpu_enhanced2)
    blurred = gpu_blurred.download()

    # Otsu threshold (CPU — no CUDA equivalent)
    _, mask = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Morphological ops on GPU
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                      (IR_MORPH_KSIZE, IR_MORPH_KSIZE))
    morph_close = cv.cuda.createMorphologyFilter(cv.MORPH_CLOSE, cv.CV_8UC1, kernel)
    morph_open = cv.cuda.createMorphologyFilter(cv.MORPH_OPEN, cv.CV_8UC1, kernel)
    gpu_mask = cv.cuda_GpuMat()
    gpu_mask.upload(mask)
    gpu_mask = morph_close.apply(gpu_mask)
    gpu_mask = morph_close.apply(gpu_mask)
    gpu_mask = morph_open.apply(gpu_mask)
    mask = gpu_mask.download()

    # Contours (CPU)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) > IR_MIN_CONTOUR_AREA]
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:6]

    mask = np.zeros(mask.shape, dtype=np.uint8)
    for c in contours:
        cv.drawContours(mask, [c], 0, 255, -1)

    gpu_mask.upload(mask)
    gauss2 = cv.cuda.createGaussianFilter(cv.CV_8UC1, cv.CV_8UC1, (5, 5), 0)
    gpu_mask = gauss2.apply(gpu_mask)
    mask = gpu_mask.download()

    segmented = cv.bitwise_and(enhanced, enhanced, mask=mask)
    result = cv.adaptiveThreshold(
        segmented, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, IR_THRESH_BLOCK, IR_THRESH_C)

    return result


# ===================================================================
# Factory
# ===================================================================

def get_processor(mode="rgb", use_cuda=False):
    """
    Return the appropriate processing function.

    Args:
        mode: "rgb" for standard cameras, "ir" for infrared cameras
        use_cuda: whether to use GPU-accelerated path
    """
    processors = {
        ("rgb", False): process_rgb_cpu,
        ("rgb", True):  process_rgb_gpu,
        ("ir",  False): process_ir_cpu,
        ("ir",  True):  process_ir_gpu,
    }
    key = (mode, use_cuda)
    if key not in processors:
        raise ValueError(f"Unknown processor mode: {mode}, cuda: {use_cuda}")
    return processors[key]
