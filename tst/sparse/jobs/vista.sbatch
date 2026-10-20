#!/usr/bin/env bash
#SBATCH --job-name=sparse-validation
#SBATCH --partition=gh,gh-dev
#SBATCH --account=AST22008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=sparse-%j.out
set -euo pipefail
bash tst/sparse/run_gpu.sh build-sparse-cuda results-vista
