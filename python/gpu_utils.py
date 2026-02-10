"""GPU availability detection and detailed device reporting."""

import cv2 as cv


def check_gpu():
    """Check which GPU backends are available and gather device info."""
    info = {
        "opencv_cuda": False,
        "dlib_cuda": False,
        "gpu_name": "None",
        "gpu_memory": "N/A",
        "cuda_version": "N/A",
        "opencv_cuda_devices": 0,
        "dlib_cuda_devices": 0,
    }

    # --- OpenCV CUDA ---
    try:
        count = cv.cuda.getCudaEnabledDeviceCount()
        info["opencv_cuda_devices"] = count
        if count > 0:
            info["opencv_cuda"] = True
            # Get device info for the first GPU
            cv.cuda.setDevice(0)
            dev = cv.cuda.DeviceInfo(0)
            info["gpu_name"] = dev.name()
            mem_bytes = dev.totalMemory()
            info["gpu_memory"] = f"{mem_bytes / (1024 ** 3):.1f} GB"
            major = dev.majorVersion()
            minor = dev.minorVersion()
            info["compute_capability"] = f"{major}.{minor}"
    except Exception:
        pass

    # --- Try nvidia-smi for driver + CUDA version ---
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                info["driver_version"] = parts[0].strip()
                # Prefer nvidia-smi name if we don't already have it
                if info["gpu_name"] == "None":
                    info["gpu_name"] = parts[1].strip()
                if info["gpu_memory"] == "N/A":
                    info["gpu_memory"] = f"{int(parts[2].strip()) / 1024:.1f} GB"

        # CUDA version
        result2 = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5)
        if result2.returncode == 0:
            for line in result2.stdout.splitlines():
                if "release" in line.lower():
                    # e.g. "Cuda compilation tools, release 12.2, V12.2.140"
                    info["cuda_version"] = line.split("release")[-1].split(",")[0].strip()
                    break
    except Exception:
        pass

    # --- dlib CUDA ---
    try:
        import dlib
        info["dlib_cuda"] = bool(dlib.DLIB_USE_CUDA)
        if info["dlib_cuda"]:
            info["dlib_cuda_devices"] = dlib.cuda.get_num_devices()
            if info["dlib_cuda_devices"] == 0:
                info["dlib_cuda"] = False
    except Exception:
        pass

    return info


def print_gpu_status(gpu_info):
    """Print detailed GPU acceleration status to console."""
    print("=" * 50)
    print("  GPU STATUS")
    print("=" * 50)
    print(f"  Device:          {gpu_info['gpu_name']}")
    print(f"  Memory:          {gpu_info['gpu_memory']}")
    if "compute_capability" in gpu_info:
        print(f"  Compute Cap:     {gpu_info['compute_capability']}")
    if "driver_version" in gpu_info:
        print(f"  Driver:          {gpu_info['driver_version']}")
    print(f"  CUDA Toolkit:    {gpu_info['cuda_version']}")
    print("-" * 50)
    cv_status = f"YES ({gpu_info['opencv_cuda_devices']} device(s))" if gpu_info["opencv_cuda"] else "NO"
    dlib_status = f"YES ({gpu_info['dlib_cuda_devices']} device(s))" if gpu_info["dlib_cuda"] else "NO"
    print(f"  OpenCV CUDA:     {cv_status}")
    print(f"  dlib CUDA:       {dlib_status}")
    print(f"  MediaPipe GPU:   Attempting (OpenGL ES 3.1+ required)")
    print("=" * 50)
