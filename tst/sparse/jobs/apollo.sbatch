#!/usr/bin/env bash
#SBATCH --job-name=sparse-validation
#SBATCH --partition=apollo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=sparse-%j.out
set -euo pipefail
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/lib64:${LD_LIBRARY_PATH:-}
bash tst/sparse/run_gpu.sh build-sparse-cuda results-apollo
