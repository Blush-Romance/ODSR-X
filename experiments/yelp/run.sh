#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$ROOT/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/dataset"}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/outputs"}
mkdir -p "$OUTPUT_DIR"
cd "$ROOT/source"

"$PYTHON_BIN" -B -u "$ROOT/train_gpu_memory.py" \
  --data_root "${DATA_ROOT}" \
  --dataset yelp2018 \
  --train \
  --device cuda \
  --finetune_role scratch \
  --seed 2026 \
  --lr 0.001 \
  --reg 0 \
  --ssl_reg 0.43 \
  --eps 0.54 \
  --dropout 0.17 \
  --batch_size 2048 \
  --max_epoch 100 \
  --eval_interval 1 \
  --patience 50 \
  --eval_batch 64 \
  --eval_item_batch 512 \
  --out "${OUTPUT_DIR}/best.pth" \
  2>&1 | tee "$OUTPUT_DIR/training.log"
