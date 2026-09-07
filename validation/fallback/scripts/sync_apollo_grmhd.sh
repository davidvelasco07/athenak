#!/usr/bin/env bash
# Sync GRMHD multi-D fallback plots/videos from Apollo into the validation tree.
# Run this in a normal local terminal (where `ssh apollo` already works).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/figures/apollo_grmhd"
LIST="${ROOT}/results/_logs/apollo_grmhd_listing.txt"
mkdir -p "$DEST" "$(dirname "$LIST")"

echo "Listing Apollo fallback runs..."
ssh apollo 'bash -s' <<'REMOTE' | tee "$LIST"
set -e
FB=~/athenak/fallback
echo "HOST=$(hostname)"
echo "=== runs/ ==="
ls -la "$FB/runs" 2>/dev/null || true
echo "=== dirs matching gr/blast/mink/mood/fb ==="
find "$FB/runs" -maxdepth 4 -type d \( \
  -iname '*gr*' -o -iname '*blast*' -o -iname '*mink*' -o -iname '*mood*' -o -iname '*fb*' \
\) 2>/dev/null | head -200
echo "=== media (mp4/gif/png) ==="
find "$FB/runs" -maxdepth 6 \( \
  -iname '*.mp4' -o -iname '*.gif' -o -iname '*.png' -o -iname '*movie*' -o -iname '*video*' \
\) 2>/dev/null | head -300
echo "=== other likely roots ==="
ls -la "$FB" 2>/dev/null | head -40
find "$FB" -maxdepth 3 -type d \( -iname '*blast*' -o -iname '*grmhd*' -o -iname '*movie*' \) 2>/dev/null | head -80
REMOTE

echo
echo "Wrote listing to $LIST"
echo "Inspect it, then re-run with SYNC=1 to copy candidate media, e.g.:"
echo "  SYNC=1 $0"
echo "Or set APOLLO_GLOB to a remote path pattern."

if [[ "${SYNC:-0}" != "1" ]]; then
  exit 0
fi

# Default: pull all png/mp4/gif under runs that look GR/blast related
REMOTE_BASE="${APOLLO_BASE:-~/athenak/fallback/runs}"
rsync -avz --progress \
  --include='*/' \
  --include='*.png' --include='*.mp4' --include='*.gif' --include='*.pdf' \
  --exclude='*' \
  "apollo:${REMOTE_BASE}/" "$DEST/"

echo "Synced into $DEST"
ls -la "$DEST" | head -40
