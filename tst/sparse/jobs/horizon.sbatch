#!/usr/bin/env bash
#SBATCH --job-name=sparse-validation
#SBATCH --partition=debug-shared
#SBATCH --account=AST26021
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=sparse-%j.out
set -euo pipefail
bash tst/sparse/run_gpu.sh build-sparse-cuda results-horizon
