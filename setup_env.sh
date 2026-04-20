#!/bin/bash
# ============================================================
# Helmet Detection System - Environment Setup Script
# Run this ONCE before anything else: bash setup_env.sh
# ============================================================

echo "=============================================="
echo "  Helmet Detection System - Environment Setup"
echo "=============================================="

# --- 1. Check Python version ---
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "[INFO] Python version: $python_version"

# --- 2. Create virtual environment ---
echo "[STEP 1] Creating virtual environment..."
python3 -m venv helmet_env
source helmet_env/bin/activate

# --- 3. Upgrade pip ---
echo "[STEP 2] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- 4. Install PyTorch (auto-detects CUDA) ---
echo "[STEP 3] Installing PyTorch..."
# Detect CUDA availability
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-)
    echo "[INFO] CUDA detected: $cuda_version"
    echo "[INFO] Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "[INFO] No CUDA detected. Installing CPU-only PyTorch..."
    echo "[INFO] For GPU support, install CUDA 11.8+ from: https://developer.nvidia.com/cuda-downloads"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# --- 5. Install all other requirements ---
echo "[STEP 4] Installing project requirements..."
pip install -r requirements.txt

# --- 6. Create .env file for Kaggle API (optional) ---
echo "[STEP 5] Setting up Kaggle (optional for dataset download)..."
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json.template << 'EOF'
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
EOF
echo "[INFO] Edit ~/.kaggle/kaggle.json with your Kaggle credentials for dataset download."

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "  Activate environment: source helmet_env/bin/activate"
echo "=============================================="
