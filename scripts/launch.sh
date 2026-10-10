#!/usr/bin/env bash
# usage: scripts/launch.sh <tag> <dataset> [odsrx.train args...]
set -euo pipefail
ROOT=/mnt/data/zzx/Code/ODSR; TAG=$1; DS=$2; shift 2
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
RUN_ID="$(hostname -s)_$(date +%Y%m%d_%H%M%S)_${DS}_${TAG}"
mkdir -p "$ROOT/runs/$RUN_ID"; cd "$ROOT"
echo "python -m odsrx.train --dataset $DS --run_dir runs/$RUN_ID $*" > "runs/$RUN_ID/cmd.txt"
nohup python -u -m odsrx.train --dataset "$DS" --run_dir "runs/$RUN_ID" "$@" > "runs/$RUN_ID/train.log" 2>&1 &
echo $! > "runs/$RUN_ID/pid.txt"; echo "$RUN_ID pid=$(cat runs/$RUN_ID/pid.txt)"
