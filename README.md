# HeteroSched

HeteroSched is an adaptive two-tier heterogeneous CPU-GPU task scheduling engine designed for unified system workloads with mixed compute and transfer characteristics. It pairs a threshold-based fast-path filter with a compile-embedded decision forest model, providing low-overhead task placement across CPU worker threads and CUDA streams.

## System Prerequisites

- **Operating System**: Linux (Ubuntu 22.04 / 24.04 LTS or WSL2 Ubuntu)
- **Compiler**: C++17 compliant compiler (`g++` or `clang++` >= 11)
- **Build Tools**: CMake >= 3.18, Make
- **GPU Toolchain**: NVIDIA CUDA Toolkit >= 12.0 with `nvcc` and compatible NVIDIA GPU
- **Python**: Python >= 3.10 with `pip` and `venv`
- **Node.js**: Node.js >= 18 and `npm`

---

## 1. Build the C++ Benchmark Engine

Compile the C++ runtime and `heterosched_bench` executable:

```bash
# Using the build script:
./scripts/build.sh

# Or manual CMake build:
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
```

The compiled binary will be placed at `build/heterosched_bench`.

---

## 2. Set Up and Start the Flask Backend

The backend provides local REST APIs to inspect hardware specs, query approved workload traces, run benchmarks, and export result CSVs.

```bash
# 1. Activate Python virtual environment:
source .venv/bin/activate
# Or create one if not yet initialized:
# python3 -m venv .venv && source .venv/bin/activate

# 2. Install backend dependencies:
pip install -r requirements.txt -r server/requirements.txt

# 3. Start the Flask API server (runs on port 5000):
python3 -m server.app
```

The backend API will be available at `http://127.0.0.1:5000`.

---

## 3. Set Up and Start the Vite Frontend Dashboard

The frontend is a React + Vite dashboard providing real-time telemetry visualizations, workload trace selectors, performance metrics, and task inspection tables.

```bash
# 1. Navigate to the frontend directory:
cd frontend

# 2. Install frontend dependencies:
npm install

# 3. Start the Vite development server (runs on port 5173):
npm run dev

# 4. (Optional) Build the production bundle:
npm run build
```

Open `http://localhost:5173` in your browser to access the interactive dashboard.
