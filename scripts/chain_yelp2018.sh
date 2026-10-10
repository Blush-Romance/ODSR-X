#!/usr/bin/env bash
# Finishes the Yelp2018 pipeline unattended: val-run -> fusion selection on val -> full run (E* epochs) -> one test evaluation.
set -uo pipefail
ROOT=/mnt/data/zzx/Code/ODSR; cd $ROOT
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
VAL=$(ls -d runs/hs-Ubuntu_20261011_013921_yelp2018_odsr-valsplit-s2026)
until grep -qE "^DONE|Traceback" $VAL/train.log; do sleep 120; done
grep -q Traceback $VAL/train.log && { echo "VAL RUN FAILED"; exit 1; }
E=$(python -c "import json;print(json.load(open(\"$VAL/result.json\"))[\"best\"][\"epoch\"])")
echo "E*=$E"
python -m odsrx.fuse --dataset yelp2018 --val_ratio 0.2 --graph $VAL/best.pth --lams 400 --ps 0,0.25 --betas 0,0.25 --ws 0,0.1,0.25,0.5,1 --threads 12 --out results/fuse/select_yelp2018.json > logs/fuse/select_yelp2018.log 2>&1
read LAM P BETA W < <(python -c "import json;b=json.load(open(\"results/fuse/select_yelp2018.json\"))[\"best\"];print(b[\"lam\"],b[\"p\"],b[\"beta\"],b[\"w\"])")
echo "selected lam=$LAM p=$P beta=$BETA w=$W"
FULL_ID=$(scripts/launch.sh odsr-full-E${E}-s2026 yelp2018 --device cpu --threads 20 --prior none --max_epoch $E --eval_interval $E | cut -d" " -f1)
until grep -qE "^DONE|Traceback" runs/$FULL_ID/train.log; do sleep 120; done
python -m odsrx.fuse --dataset yelp2018 --graph runs/$FULL_ID/best.pth --lams $LAM --ps $P --betas $BETA --w $W --threads 20 --out results/fuse/apply_yelp2018_own.json > logs/fuse/apply_yelp2018_own.log 2>&1
python -m odsrx.fuse --dataset yelp2018 --graph ../ODSR_official/checkpoints/Yelp2018.pth --lams $LAM --ps $P --betas $BETA --w $W --threads 20 --out results/fuse/apply_yelp2018_official.json > logs/fuse/apply_yelp2018_official.log 2>&1
echo CHAIN_YELP_DONE
