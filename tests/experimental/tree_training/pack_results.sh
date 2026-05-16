#!/usr/bin/env bash
# Collect Phase J/K results into a single tarball for transfer back to local.
#
# Run AFTER Phase J + K finish. Output goes to ``/tmp/magi_v1_results_*.tgz``.
#
# Usage:
#   bash tests/experimental/tree_training/pack_results.sh

set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUT="/tmp/magi_v1_results_${TS}.tgz"

# Files we care about. Use ``ls`` to expand globs, ignore missing files.
shopt -s nullglob
FILES=(
  /tmp/magi_speedup_*.csv
  /tmp/magi_speedup_*.png
  /tmp/magi_phase_j_*.log
  /tmp/phase_k_bench.log
)
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ERROR: no result files found in /tmp. Did Phase J / K actually run?"
  exit 1
fi

echo "Packaging ${#FILES[@]} file(s) into ${OUT} ..."
for f in "${FILES[@]}"; do
  echo "  + ${f}"
done

tar -czf "$OUT" "${FILES[@]}"
ls -lh "$OUT"

echo ""
echo "On your local Mac, pull with:"
echo "  scp -P <SSH_PORT> -i ~/.ssh/id_ed25519 root@<POD_IP>:${OUT} ~/Desktop/"
