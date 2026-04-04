#!/bin/bash
# setup_env.sh — 一键搭建 verl 0.5B GRPO 训练环境
# 用法: cd /root/verl && bash setup_env.sh
set -euo pipefail

echo "========== [0/5] 安装系统工具 =========="
command -v tmux &>/dev/null || { apt-get update && apt-get install -y tmux; }

echo "========== [1/5] 升级 PyTorch (需要 2.5+) =========="
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "========== [2/5] 安装 vLLM =========="
pip install vllm

echo "========== [3/5] 安装 verl 及依赖 =========="
cd /root/verl
pip install -e .
# flash-attn 编译较慢，但对性能很重要
python3 -c "import flash_attn" 2>/dev/null || MAX_JOBS=4 pip install flash-attn --no-build-isolation

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

from torch.distributed.tensor import DTensor
print('DTensor: OK')

import vllm
print(f'vLLM: {vllm.__version__}')

import verl
print('verl: OK')

import os
assert os.path.exists(os.path.expanduser('~/data/gsm8k/train.parquet')), 'GSM8K data missing!'
print('GSM8K data: OK')

print('\\n=== All checks passed! ===')
print('Run training with:')
print('  bash examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh')
"
