#!/usr/bin/env bash
# Faithful re-run of the published Douban-Book ODSR command, with a unique run ID.
set -euo pipefail
ROOT=/mnt/data/zzx/Code/ODSR
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data/zzx/conda_envs/odsr
GPU=${GPU:-0}; SEED=${SEED:-2026}
RUN_ID="$(hostname -s)_$(date +%Y%m%d_%H%M%S)_douban_odsr-repro_s${SEED}"
mkdir -p "$ROOT/logs/$RUN_ID" "$ROOT/checkpoints/$RUN_ID"
{ echo "server=$(hostname)"; echo "gpu=$GPU"; echo "commit=$(git -C $ROOT rev-parse HEAD)"; echo "start=$(date -Is)"; } > "$ROOT/logs/$RUN_ID/meta.txt"
cd "$ROOT/source"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python -B -u main.py \
  --data_root "$ROOT/dataset" --dataset douban-book --train --device cuda --seed $SEED \
  --lr 0.002 --reg 1e-5 --ssl_reg 0.5 --eps 0.8 --batch_size 2048 --max_epoch 100 \
  --eval_interval 1 --patience 50 --eval_batch 64 --eval_item_batch 512 \
  --out "$ROOT/checkpoints/$RUN_ID/best.pth" > "$ROOT/logs/$RUN_ID/train.log" 2>&1 &
echo $! | tee "$ROOT/logs/$RUN_ID/pid.txt"; echo "$RUN_ID"
