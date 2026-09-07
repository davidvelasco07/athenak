#!/usr/bin/env bash
# Inventory + optional sync of ~/balsara_jet (and name variants) from Apollo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIST="${ROOT}/results/_logs/apollo_balsara_jet_listing.txt"
DEST="${ROOT}/figures/apollo_grmhd"
mkdir -p "$DEST" "$(dirname "$LIST")"

ssh apollo 'bash -s' <<'REMOTE' | tee "$LIST"
set -e
echo "HOST=$(hostname)"
echo "=== home matches ==="
ls -lad ~/balsara_jet ~/Balsara_jet ~/*balsara* ~/*jet* 2>/dev/null || true
echo "=== tree (depth 3) ==="
for d in ~/balsara_jet ~/Balsara_jet; do
  if [ -d "$d" ]; then
    echo "-- $d"
    find "$d" -maxdepth 3 \( -type d -o -iname '*.png' -o -iname '*.mp4' -o -iname '*.gif' -o -iname '*.pdf' -o -iname '*.athinput' -o -iname '*mood*' \) 2>/dev/null | head -250
  fi
done
echo "=== media under balsara/jet ==="
find ~ -maxdepth 5 \( -path '*balsara*' -o -path '*Balsara*' -o -path '*jet*' \) \( \
  -iname '*.png' -o -iname '*.mp4' -o -iname '*.gif' -o -iname '*.pdf' \
\) 2>/dev/null | head -200
REMOTE

echo "Wrote $LIST"
if [[ "${SYNC:-0}" == "1" ]]; then
  REMOTE_DIR="${APOLLO_PATH:-~/balsara_jet}"
  echo "Syncing apollo:$REMOTE_DIR -> $DEST"
  rsync -avz --progress \
    --include='*/' \
    --include='*.png' --include='*.mp4' --include='*.gif' --include='*.pdf' \
    --include='*.athinput' --include='*.log' --include='*summary*' \
    --exclude='*' \
    "apollo:${REMOTE_DIR}/" "$DEST/"
  ls -la "$DEST" | head -50
fi
