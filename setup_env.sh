#!/bin/bash
# setup_env.sh — 一键搭建 verl 0.5B GRPO 训练环境（可重入）
# 用法: cd /root/verl && bash setup_env.sh
# 参考: https://www.cnblogs.com/rh-li/p/19302501
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV_NAME="verl"

echo "========== [0/7] 安装系统工具 =========="
command -v tmux &>/dev/null || { apt-get update && apt-get install -y tmux; }

echo "========== [1/7] 安装 Miniconda =========="
if [ ! -d "$REPO_ROOT/../miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$REPO_ROOT/../miniconda3"
    rm -f /tmp/miniconda.sh
fi
eval "$("$REPO_ROOT/../miniconda3/bin/conda" shell.bash hook)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "========== [2/7] 创建 conda 虚拟环境 =========="
if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    conda create -n "$CONDA_ENV_NAME" python=3.12 -y
fi
conda activate "$CONDA_ENV_NAME"
echo "Python: $(python3 --version) at $(which python3)"

echo "========== [3/7] 安装 PyTorch + vLLM + 基础依赖（锁定版本） =========="
# 如果系统已有 torch，直接复用（避免重复下载 766MB）
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch 已安装: $(python3 -c 'import torch; print(torch.__version__)'), 跳过"
else
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
fi
pip install vllm==0.8.2
pip install ray==2.44.0 tensordict==0.6.2
pip install transformers accelerate datasets peft hydra-core wandb

echo "========== [4/7] 安装 flash-attn（预编译 wheel） =========="
if ! python3 -c "import importlib.metadata; importlib.metadata.version('flash_attn')" 2>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    WHEEL="flash_attn-2.7.4+cu12torch2.6cxx11abiFALSE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/${WHEEL}"
    echo "下载预编译 wheel: ${WHEEL}"
    wget -nv "${WHEEL_URL}" && pip install --no-cache-dir "${WHEEL}" && rm -f "${WHEEL}" \
        || { echo "预编译 wheel 不可用，源码编译..."; MAX_JOBS=8 pip install flash-attn --no-build-isolation; }
fi

echo "========== [5/7] 安装 verl =========="
cd "$REPO_ROOT"
pip install --no-deps -e .

echo "========== [6/7] 准备 GSM8K 数据 =========="
if [ ! -f ~/data/gsm8k/train.parquet ]; then
    mkdir -p ~/data/gsm8k
    python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
else
    echo "GSM8K 数据已存在，跳过"
fi

echo "========== [7/7] 验证环境 =========="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')

import vllm
print(f'vLLM: {vllm.__version__}')

import verl
print('verl: OK')

import importlib.metadata
try:
    fa_ver = importlib.metadata.version('flash_attn')
    print(f'flash-attn: {fa_ver}')
except importlib.metadata.PackageNotFoundError:
    print('flash-attn: NOT INSTALLED')

import os
assert os.path.exists(os.path.expanduser('~/data/gsm8k/train.parquet')), 'GSM8K data missing!'
print('GSM8K data: OK')

print('\n=== All checks passed! ===')
"

echo -e "
${BOLD}${GREEN}========================================${RESET}
${BOLD}${GREEN}  Setup complete!${RESET}
${BOLD}${GREEN}========================================${RESET}

${BOLD}${YELLOW}⚠  每个新终端需要先执行:${RESET}
${GREEN}   source $REPO_ROOT/../miniconda3/bin/activate && conda activate $CONDA_ENV_NAME${RESET}

${YELLOW}# 启动 tmux:${RESET}
${GREEN}   tmux new -s verl${RESET}

${YELLOW}# 开始训练:${RESET}
${GREEN}   bash examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh${RESET}
"
