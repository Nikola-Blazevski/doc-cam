#!/bin/bash
set -e

# ============================================================
# Hand & Face Detection Tracker — Full Install Script
# ============================================================
# Installs system packages, Python dependencies, ML models,
# and optionally GPU-accelerated builds of dlib.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh           # standard install (CPU)
#   ./install.sh --gpu     # attempt GPU-accelerated dlib
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

GPU_MODE=false
if [[ "$1" == "--gpu" ]]; then
    GPU_MODE=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $1"; }

check_command() {
    if command -v "$1" &>/dev/null; then
        success "$1 found: $(command -v "$1")"
        return 0
    else
        fail "$1 not found"
        return 1
    fi
}

# -----------------------------------------------------------
# 1. Detect OS
# -----------------------------------------------------------
echo ""
echo "============================================"
echo "  Hand & Face Tracker — Installer"
echo "============================================"
echo ""

info "Detecting operating system..."

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS_ID="$ID"
    OS_VERSION="$VERSION_ID"
    info "Detected: $PRETTY_NAME"
elif [[ "$(uname)" == "Darwin" ]]; then
    OS_ID="macos"
    OS_VERSION="$(sw_vers -productVersion)"
    info "Detected: macOS $OS_VERSION"
else
    OS_ID="unknown"
    warn "Could not detect OS. Will attempt generic install."
fi

# -----------------------------------------------------------
# 2. Install system dependencies
# -----------------------------------------------------------
echo ""
info "Installing system dependencies..."

install_system_deps_apt() {
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 \
        python3-pip \
        python3-venv \
        cmake \
        build-essential \
        pkg-config \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libopenblas-dev \
        liblapack-dev \
        libx11-dev \
        libgtk-3-dev \
        libboost-all-dev
    success "System packages installed (apt)"
}

install_system_deps_dnf() {
    sudo dnf install -y -q \
        python3 \
        python3-pip \
        cmake \
        gcc-c++ \
        make \
        wget \
        mesa-libGL \
        glib2 \
        openblas-devel \
        lapack-devel \
        gtk3-devel \
        boost-devel \
        libX11-devel \
        libXext-devel \
        libXrender-devel
    success "System packages installed (dnf)"
}

install_system_deps_pacman() {
    sudo pacman -Sy --noconfirm --needed \
        python \
        python-pip \
        cmake \
        base-devel \
        wget \
        mesa \
        glib2 \
        openblas \
        lapack \
        gtk3 \
        boost \
        libx11 \
        libxext \
        libxrender
    success "System packages installed (pacman)"
}

install_system_deps_brew() {
    brew install python cmake wget boost openblas
    success "System packages installed (brew)"
}

case "$OS_ID" in
    ubuntu|debian|pop|linuxmint|elementary)
        install_system_deps_apt
        ;;
    fedora|rhel|centos|rocky|almalinux)
        install_system_deps_dnf
        ;;
    arch|manjaro|endeavouros)
        install_system_deps_pacman
        ;;
    macos)
        if check_command brew; then
            install_system_deps_brew
        else
            warn "Homebrew not found. Install it from https://brew.sh"
            warn "Then re-run this script."
            exit 1
        fi
        ;;
    *)
        warn "Unsupported OS. Please install manually:"
        warn "  python3, pip, cmake, build tools, wget, OpenGL libs, boost"
        ;;
esac

# -----------------------------------------------------------
# 3. Create virtual environment
# -----------------------------------------------------------
echo ""
VENV_DIR="$SCRIPT_DIR/venv"

if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
info "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip setuptools wheel -q
success "pip upgraded"

# -----------------------------------------------------------
# 4. Install Python dependencies
# -----------------------------------------------------------
echo ""
info "Installing Python packages..."

pip install -q numpy
success "numpy"

pip install -q opencv-contrib-python
success "opencv-contrib-python"

pip install -q "mediapipe==0.10.31"
success "mediapipe 0.10.31"

pip install -q cmake
success "cmake (Python)"

# -----------------------------------------------------------
# 5. Install dlib + face_recognition
# -----------------------------------------------------------
echo ""
info "Installing dlib and face_recognition..."

if $GPU_MODE; then
    info "GPU mode requested — checking for CUDA..."

    if command -v nvcc &>/dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
        success "CUDA found: $CUDA_VERSION"
        info "Building dlib with CUDA support (this may take several minutes)..."

        pip install -q dlib \
            --config-settings="--build-option=--yes" \
            --config-settings="--build-option=DLIB_USE_CUDA" \
            --no-cache-dir 2>&1 | tail -5

        if python3 -c "import dlib; assert dlib.DLIB_USE_CUDA" 2>/dev/null; then
            success "dlib installed with CUDA support"
        else
            warn "dlib installed but CUDA not active — falling back to CPU"
            pip install -q --force-reinstall dlib
        fi
    else
        warn "nvcc not found — CUDA not available"
        warn "Installing dlib without CUDA (CPU only)"
        pip install -q dlib
    fi
else
    info "Installing dlib (CPU)..."
    pip install -q dlib
    success "dlib (CPU)"
fi

pip install -q face_recognition
success "face_recognition"

# -----------------------------------------------------------
# 6. Download ML models
# -----------------------------------------------------------
echo ""
info "Downloading ML models..."

HAND_MODEL="$SCRIPT_DIR/hand_landmarker.task"
FACE_MODEL="$SCRIPT_DIR/blaze_face_short_range.tflite"
POSE_MODEL="$SCRIPT_DIR/pose_landmarker_lite.task"

HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
POSE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if [[ -f "$HAND_MODEL" ]]; then
    info "Hand model already exists, skipping download"
else
    wget -q --show-progress -O "$HAND_MODEL" "$HAND_URL"
fi
success "hand_landmarker.task ($(du -h "$HAND_MODEL" | cut -f1))"

if [[ -f "$FACE_MODEL" ]]; then
    info "Face model already exists, skipping download"
else
    wget -q --show-progress -O "$FACE_MODEL" "$FACE_URL"
fi
success "blaze_face_short_range.tflite ($(du -h "$FACE_MODEL" | cut -f1))"

if [[ -f "$POSE_MODEL" ]]; then
    info "Pose model already exists, skipping download"
else
    wget -q --show-progress -O "$POSE_MODEL" "$POSE_URL"
fi
success "pose_landmarker_lite.task ($(du -h "$POSE_MODEL" | cut -f1))"

# -----------------------------------------------------------
# 7. Verify installation
# -----------------------------------------------------------
echo ""
info "Verifying installation..."

python3 -c "
import sys

errors = []

try:
    import cv2
    print(f'  opencv:           {cv2.__version__}')
    try:
        n = cv2.cuda.getCudaEnabledDeviceCount()
        print(f'  opencv CUDA:      {n} device(s)')
    except:
        print(f'  opencv CUDA:      not available')
except ImportError as e:
    errors.append(f'opencv: {e}')

try:
    import mediapipe as mp
    print(f'  mediapipe:        {mp.__version__}')
except ImportError as e:
    errors.append(f'mediapipe: {e}')

try:
    import numpy as np
    print(f'  numpy:            {np.__version__}')
except ImportError as e:
    errors.append(f'numpy: {e}')

try:
    import dlib
    cuda_status = 'YES' if dlib.DLIB_USE_CUDA else 'NO'
    print(f'  dlib:             {dlib.__version__} (CUDA: {cuda_status})')
except ImportError as e:
    errors.append(f'dlib: {e}')

try:
    import face_recognition
    print(f'  face_recognition: installed')
except ImportError as e:
    errors.append(f'face_recognition: {e}')

if errors:
    print()
    for err in errors:
        print(f'  ERROR: {err}')
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    success "All packages verified"
else
    fail "Some packages failed to install. See errors above."
    exit 1
fi

# -----------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------
echo ""
echo "============================================"
echo -e "  ${GREEN}Installation complete!${NC}"
echo "============================================"
echo ""
echo "  To run:"
echo ""
echo "    cd $SCRIPT_DIR"
echo "    source venv/bin/activate"
echo "    python main.py 0           # webcam"
echo "    python main.py video.mp4   # video file"
echo ""
echo "  To deactivate the virtual environment:"
echo "    deactivate"
echo ""
if $GPU_MODE; then
    echo "  GPU mode was requested."
    echo "  Check the output above to confirm CUDA is active."
    echo ""
fi
