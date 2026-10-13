#!/usr/bin/env bash
set -euo pipefail

echo "Setting up HeteroSched development environment..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is required but not found."
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. CUDA compilation will fail."
    echo "  Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
fi

if ! command -v cmake &> /dev/null; then
    echo "WARNING: cmake not found. Install cmake >= 3.18."
fi

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Generating ML dataset..."
python3 ml/dataset_generator.py

echo "Training scheduling model..."
python3 ml/train_model.py

echo "Exporting model to C++ format..."
python3 ml/export_to_cpp.py

echo "Generating benchmark traces..."
mkdir -p benchmarks/traces
python3 benchmarks/trace_generator.py

echo ""
echo "Environment setup complete."
echo "  Run 'scripts/build.sh' to compile the C++ runtime."
