#!/usr/bin/env bash
# Broader Apollo search for GRMHD / blast / multi-D fallback artifacts.
# Run in a local terminal where `ssh apollo` works.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIST="${ROOT}/results/_logs/apollo_grmhd_listing.txt"
DEST="${ROOT}/figures/apollo_grmhd"
mkdir -p "$DEST" "$(dirname "$LIST")"

ssh apollo 'bash -s' <<'REMOTE' | tee "$LIST"
set -e
echo "HOST=$(hostname)"
echo "=== fallback top ==="
ls ~/athenak/fallback | head -60

echo "=== build_* dirs ==="
ls -ld ~/athenak/fallback/build* 2>/dev/null || true

echo "=== find blast/grmhd/minkowski/movie under athenak (depth 5) ==="
find ~/athenak -maxdepth 5 \( \
  -iname '*blast*' -o -iname '*grmhd*' -o -iname '*minkowski*' \
  -o -iname '*movie*' -o -iname '*video*' -o -iname '*mood*gr*' \
\) 2>/dev/null | head -200

echo "=== png/mp4/gif under fallback (not only runs/) ==="
find ~/athenak/fallback -maxdepth 6 \( \
  -iname '*.png' -o -iname '*.mp4' -o -iname '*.gif' -o -iname '*.pdf' \
\) 2>/dev/null | head -300

echo "=== home-level run-like dirs ==="
ls -ld ~/runs ~/scratch ~/work ~/data ~/Movies ~/Desktop 2>/dev/null || true
find ~ -maxdepth 3 -type d \( -iname '*blast*' -o -iname '*grmhd*' -o -iname '*fallback*' \) 2>/dev/null | head -80

echo "=== recent png/mp4 anywhere under home (depth 4, last 180d) ==="
find ~ -maxdepth 4 \( -iname '*.mp4' -o -iname '*blast*.png' -o -iname '*grmhd*.png' \) \
  -mtime -180 2>/dev/null | head -100
REMOTE

echo
echo "Wrote $LIST"
if [[ "${SYNC:-0}" == "1" && -n "${APOLLO_PATH:-}" ]]; then
  echo "Syncing apollo:$APOLLO_PATH -> $DEST"
  rsync -avz "apollo:$APOLLO_PATH" "$DEST/"
fi
