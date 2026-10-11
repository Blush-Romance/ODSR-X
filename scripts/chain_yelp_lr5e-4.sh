#!/usr/bin/env bash
# Yelp lr 5e-4 (val-selected E*=6): 3 full-train seeds in parallel, then select (w, w_soc) on val with the val graph and apply once on test.
cd /mnt/data/zzx/Code/ODSR
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
IDS=()
for S in 2026 2027 2028; do IDS+=($(scripts/launch.sh odsr-full-lr5e-4-E6-s$S yelp2018 --device cpu --threads 6 --seed $S --prior none --lr 5e-4 --max_epoch 6 --eval_interval 6 | cut -d" " -f1)); sleep 2; done
for ID in "${IDS[@]}"; do until grep -qE "^DONE|Traceback" runs/$ID/train.log; do sleep 120; done; done
G=(); for S in 2026 2027 2028; do G+=(own_lr5e-4_s$S=$(ls -d runs/*yelp2018_odsr-full-lr5e-4-E6-s$S)/best.pth); done
python -m odsrx.fuse --dataset yelp2018 --val_ratio 0.2 --graph runs/hs-Ubuntu_20261011_112648_yelp2018_odsr-val-lr5e-4-s2026/best.pth --lams 400 --ps 0.25 --betas 0 --ws 0.1,0.25,0.5 --soc_ws 0,0.05,0.1,0.2 --threads 12 --out results/fuse/select_yelp2018_lr5e-4_soc.json > logs/fuse/select_yelp2018_lr5e-4_soc.log 2>&1 || { echo FAILED; exit 1; }
read W WS < <(python -c "import json;b=json.load(open(\"results/fuse/select_yelp2018_lr5e-4_soc.json\"))[\"best\"];print(b[\"w\"],b[\"w_soc\"])")
echo "selected w=$W w_soc=$WS"
for spec in "${G[@]}"; do TAG=${spec%%=*}; GP=${spec#*=}
  python -m odsrx.fuse --dataset yelp2018 --graph $GP --lams 400 --ps 0.25 --betas 0 --w $W --soc_ws $WS --threads 12 --out results/fuse/apply_yelp2018_soc_$TAG.json > logs/fuse/apply_yelp2018_soc_$TAG.log 2>&1
done
echo YELP_LR_DONE
