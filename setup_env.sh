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

# 检查 Python 模块是否已安装
installed() { python3 -c "import $1" 2>/dev/null; }
# 检查 pip 包元数据是否存在
has_pkg() { python3 -c "import importlib.metadata; importlib.metadata.version('$1')" 2>/dev/null; }

echo "========== [0/5] 安装系统工具 =========="
command -v tmux &>/dev/null || { apt-get update && apt-get install -y tmux; }

echo "========== [1/5] 安装 PyTorch + vLLM + 基础依赖（锁定版本） =========="
installed torch  && echo "torch 已安装，跳过"       || pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
installed vllm   && echo "vllm 已安装，跳过"        || pip install vllm==0.8.2
installed ray    && echo "ray 已安装，跳过"          || pip install ray==2.44.0
installed tensordict && echo "tensordict 已安装，跳过" || pip install tensordict==0.6.2
installed transformers && echo "基础依赖已安装，跳过"  || pip install transformers accelerate datasets peft hydra-core wandb

echo "========== [2/5] 安装 flash-attn（预编译 wheel） =========="
if has_pkg flash_attn; then
    echo "flash-attn 已安装，跳过"
else
    PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0].rsplit('.',1)[0])")
    WHEEL="flash_attn-2.7.3+cu12torch${TORCH_VER}cxx11abiFALSE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/${WHEEL}"
    echo "下载预编译 wheel: ${WHEEL}"
    wget -nv "${WHEEL_URL}" && pip install --no-cache-dir "${WHEEL}" && rm -f "${WHEEL}" \
        || { echo "预编译 wheel 不可用，源码编译..."; MAX_JOBS=8 pip install flash-attn --no-build-isolation; }
fi

echo "========== [3/5] 安装 verl =========="
cd "$REPO_ROOT"
installed verl && echo "verl 已安装，跳过" || pip install --no-deps -e .

echo "========== [4/5] 准备 GSM8K 数据 =========="
if [ ! -f ~/data/gsm8k/train.parquet ]; then
    mkdir -p ~/data/gsm8k
    python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
else
    echo "GSM8K 数据已存在，跳过"
fi

echo "========== [5/5] 验证环境 =========="
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

${YELLOW}# 启动 tmux:${RESET}
${GREEN}   tmux new -s verl${RESET}

${YELLOW}# 开始训练:${RESET}
${GREEN}   bash examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh${RESET}
"
