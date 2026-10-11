#!/usr/bin/env bash
# Select (w, w_soc) on validation with the val-split graph, then apply once on test for each given graph.
# usage: chain_social.sh <dataset> <val-graph> <lam> <p> <beta> <ws> <soc_ws> <threads> <tag=graph.pth>...
cd /mnt/data/zzx/Code/ODSR
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
DS=$1 VG=$2 LAM=$3 P=$4 BETA=$5 WS=$6 SWS=$7 TH=$8; shift 8
python -m odsrx.fuse --dataset $DS --val_ratio 0.2 --graph $VG --lams $LAM --ps $P --betas $BETA --ws $WS --soc_ws $SWS --threads $TH --out results/fuse/select_${DS}_soc.json > logs/fuse/select_${DS}_soc.log 2>&1 || { echo FAILED; exit 1; }
read W WSOC < <(python -c "import json;b=json.load(open(\"results/fuse/select_${DS}_soc.json\"))[\"best\"];print(b[\"w\"],b[\"w_soc\"])")
echo "selected w=$W w_soc=$WSOC"
for spec in "$@"; do
  TAG=${spec%%=*}; G=${spec#*=}
  python -m odsrx.fuse --dataset $DS --graph $G --lams $LAM --ps $P --betas $BETA --w $W --soc_ws $WSOC --threads $TH --out results/fuse/apply_${DS}_soc_$TAG.json > logs/fuse/apply_${DS}_soc_$TAG.log 2>&1
done
echo CHAIN_SOC_DONE
