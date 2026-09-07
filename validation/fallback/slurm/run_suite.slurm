#!/bin/bash
#SBATCH -J mood_val
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -p gh
#SBATCH -A AST22008

# Portable cluster launcher for expensive validation cases.
# Override ATHENA / SUITE / SCHEMES via environment.

set -euo pipefail

REPO="${REPO:-$HOME/athenak-fallback-validation}"
ATHENA="${ATHENA:-$REPO/build/src/athena}"
SUITE="${SUITE:-stress}"
SCHEMES="${SCHEMES:-ppm_fb,plm,wenoz,teno}"

cd "$REPO"
python3 validation/fallback/scripts/run_suite.py \
  --athena "$ATHENA" \
  --suite "$SUITE" \
  --schemes "$SCHEMES"
