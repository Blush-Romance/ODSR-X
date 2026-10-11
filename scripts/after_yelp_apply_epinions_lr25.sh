#!/usr/bin/env bash
# Run after the Yelp social chain (memory): apply the val-selected (w, w_soc) to the 3 lr-tuned Epinions graphs, test once.
cd /mnt/data/zzx/Code/ODSR
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
until grep -qE "CHAIN_SOC_DONE|FAILED" logs/chain_social_yelp2018.log; do sleep 120; done
read W WS < <(python -c "import json;b=json.load(open(\"results/fuse/select_epinions_lr25_soc.json\"))[\"best\"];print(b[\"w\"],b[\"w_soc\"])")
echo "selected w=$W w_soc=$WS"
for S in 2026 2027 2028; do
  G=$(ls -d runs/*odsr-full-lr2.5e-4-E16-s$S)/best.pth
  python -m odsrx.fuse --dataset epinions --graph $G --lams 200 --ps 0.25 --betas 0.25 --w $W --soc_ws $WS --threads 8 --out results/fuse/apply_epinions_lr25_soc_s$S.json > logs/fuse/apply_epinions_lr25_soc_s$S.log 2>&1
done
echo EPI_LR25_DONE
