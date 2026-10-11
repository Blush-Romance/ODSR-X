#!/usr/bin/env bash
# Epinions: select (w, w_soc) on validation, then apply once on test for the 3 own-graph seeds.
cd /mnt/data/zzx/Code/ODSR
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
python -m odsrx.fuse --dataset epinions --val_ratio 0.2 --graph runs/hs-Ubuntu_20261010_232530_epinions_odsr-valsplit-s2026/best.pth --lams 200 --ps 0.25 --betas 0.25 --ws 0.1,0.25,0.5 --soc_ws 0,0.05,0.1,0.15,0.2,0.3 --threads 6 --out results/fuse/select_epinions_soc.json > logs/fuse/select_epinions_soc.log 2>&1
read W WS < <(python -c "import json;b=json.load(open(\"results/fuse/select_epinions_soc.json\"))[\"best\"];print(b[\"w\"],b[\"w_soc\"])")
echo "selected w=$W w_soc=$WS"
for S in 2026 2027 2028; do
  G=$(ls -d runs/*epinions_odsr-full-E4-s$S | head -1)/best.pth
  python -m odsrx.fuse --dataset epinions --graph $G --lams 200 --ps 0.25 --betas 0.25 --w $W --soc_ws $WS --threads 6 --out results/fuse/apply_epinions_soc_own_s$S.json > logs/fuse/apply_epinions_soc_own_s$S.log 2>&1
done
python -m odsrx.fuse --dataset epinions --graph ../ODSR_official/checkpoints/Epinions.pth --lams 200 --ps 0.25 --betas 0.25 --w $W --soc_ws $WS --threads 6 --out results/fuse/apply_epinions_soc_official.json > logs/fuse/apply_epinions_soc_official.log 2>&1
echo CHAIN_SOC_DONE
