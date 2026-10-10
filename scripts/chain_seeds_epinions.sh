#!/usr/bin/env bash
# Extra seeds for Epinions with the configuration frozen on validation (E*=4, lam=200 p=0.25 beta=0.25 w=0.25).
cd /mnt/data/zzx/Code/ODSR
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
for S in 2027 2028; do
  ID=$(scripts/launch.sh odsr-full-E4-s$S epinions --device cpu --threads 8 --seed $S --prior none --max_epoch 4 --eval_interval 4 | cut -d" " -f1)
  until grep -qE "^DONE|Traceback" runs/$ID/train.log; do sleep 60; done
  python -m odsrx.fuse --dataset epinions --graph runs/$ID/best.pth --lams 200 --ps 0.25 --betas 0.25 --w 0.25 --threads 8 --out results/fuse/apply_epinions_own_s$S.json > logs/fuse/apply_epinions_own_s$S.log 2>&1
done
echo SEEDS_DONE
