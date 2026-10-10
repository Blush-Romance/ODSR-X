#!/usr/bin/env bash
# Extra seeds with a configuration frozen on validation.
# usage: chain_seeds.sh <dataset> <E*> <lam> <p> <beta> <w> <threads> <seed>...
cd /mnt/data/zzx/Code/ODSR
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
DS=$1 E=$2 LAM=$3 P=$4 BETA=$5 W=$6 TH=$7; shift 7
for S in "$@"; do
  ID=$(scripts/launch.sh odsr-full-E$E-s$S $DS --device cpu --threads $TH --seed $S --prior none --max_epoch $E --eval_interval $E | cut -d" " -f1)
  until grep -qE "^DONE|Traceback" runs/$ID/train.log; do sleep 60; done
  python -m odsrx.fuse --dataset $DS --graph runs/$ID/best.pth --lams $LAM --ps $P --betas $BETA --w $W --threads $TH --out results/fuse/apply_${DS}_own_s$S.json > logs/fuse/apply_${DS}_own_s$S.log 2>&1
done
echo SEEDS_DONE
