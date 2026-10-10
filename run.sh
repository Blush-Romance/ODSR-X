#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
mkdir -p "$ROOT/outputs"
cd "$ROOT/source"

"$PYTHON_BIN" -B -u main.py \
  --data_root "$ROOT/dataset" \
  --dataset douban-book \
  --train \
  --device cuda \
  --seed 2026 \
  --lr 0.002 \
  --reg 1e-5 \
  --ssl_reg 0.5 \
  --eps 0.8 \
  --batch_size 2048 \
  --max_epoch 100 \
  --eval_interval 1 \
  --patience 50 \
  --eval_batch 64 \
  --eval_item_batch 512 \
  --out "$ROOT/outputs/best.pth" \
  2>&1 | tee "$ROOT/outputs/training.log"
