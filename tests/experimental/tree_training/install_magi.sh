#!/usr/bin/env bash
# Build + install MagiAttention from source for Hopper (H100/H200).
#
# Per the official install docs:
#   https://sandai-org.github.io/MagiAttention/docs/main/user_guide/install.html
#
# magi_attention is NOT on PyPI. The package must be compiled from source.
# Build time on H200/H100: 10-20 minutes (CUTLASS DSL + nvshmem + custom CUDA kernels).
#
# Usage (inside an active conda/venv with torch + nvcc available):
#   bash tests/experimental/tree_training/install_magi.sh

set -euo pipefail

MAGI_DIR="${MAGI_DIR:-/root/MagiAttention}"
MAGI_REPO="${MAGI_REPO:-https://github.com/SandAI-org/MagiAttention.git}"
MAGI_BRANCH="${MAGI_BRANCH:-main}"

echo "=== MagiAttention install script ==="
echo "Target dir: ${MAGI_DIR}"
echo "Repo:       ${MAGI_REPO} @ ${MAGI_BRANCH}"

# 1. Verify environment
python -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"
if ! command -v nvcc &>/dev/null; then
  echo "ERROR: nvcc not found. MagiAttention requires CUDA toolkit to build."
  exit 1
fi
echo "nvcc: $(nvcc --version | grep release)"

# 2. Clone with submodules (CUTLASS + Magi's flash-attention fork are submodules)
if [[ -d "${MAGI_DIR}" ]]; then
  echo "[skip clone] ${MAGI_DIR} already exists; pulling latest"
  cd "${MAGI_DIR}"
  git fetch origin "${MAGI_BRANCH}"
  git checkout "${MAGI_BRANCH}"
  git pull origin "${MAGI_BRANCH}"
  git submodule update --init --recursive
else
  git clone --branch "${MAGI_BRANCH}" "${MAGI_REPO}" "${MAGI_DIR}"
  cd "${MAGI_DIR}"
  git submodule update --init --recursive
fi

# 3. Install Python deps (nvshmem, cutlass-dsl, einops, ...)
echo ""
echo "=== Installing requirements.txt ==="
pip install -r requirements.txt

# 4. Compile + install (Hopper path; no special env vars needed)
echo ""
echo "=== Compiling magi_attention (10-20 min on H100/H200) ==="
pip install --no-build-isolation .

# 5. Verify
echo ""
echo "=== Verifying import ==="
python - <<'EOF'
from magi_attention.api import (
    flex_flash_attn_func, magi_attn_flex_key, calc_attn,
    AttnRanges, AttnMaskType, DistAttnConfig,
    compute_pad_size, dispatch, undispatch, get_most_recent_key,
)
import magi_attention
print(f"magi_attention {magi_attention.__version__} — all key symbols importable")
EOF

echo ""
echo "=== MagiAttention install DONE ==="
