#!/usr/bin/env bash
# Same-protocol baseline: (1) train on fit split, select epoch E* on validation (method's own N@20 criterion);
# (2) retrain on full train for exactly E* epochs; (3) score test once with odsrx.evaluate.
# usage: baselines/run_baseline.sh <sgil|gbsr> <douban-book|epinions|yelp2018> <tf_threads> <seed> [VAL_RUN_DIR]
set -uo pipefail
ROOT=/mnt/data/zzx/Code/ODSR; M=$1; DS=$2; TH=$3; SEED=$4; VAL=${5:-}
cd $ROOT
case $DS in douban-book) OD=douban_book;; epinions) OD=epinions;; yelp2018) OD=yelp;; esac
case $M:$OD in
  sgil:douban_book) HP="--penalty_coff 0.15 --adv_bs 20";; sgil:yelp) HP="--penalty_coff 0.05 --adv_bs 20";; sgil:epinions) HP="--penalty_coff 0.10 --adv_bs 3";;
  gbsr:douban_book) HP="--beta 40 --sigma 2.5";;           gbsr:yelp) HP="--beta 2.0 --sigma 0.25";;          gbsr:epinions) HP="--beta 3.0 --sigma 0.25";;
esac
SCRIPT=$([ $M = sgil ] && echo run_SGIL.py || echo run_GBSR.py)
read NU NI < <(python3 -c "import json;m=json.load(open('baselines/data/$DS/meta.json'))['full'];print(m['num_user'],m['num_item'])")
run() {  # $1 tag  $2 split  $3 fixed_epochs
  local ID="$(hostname -s)_$(date +%Y%m%d_%H%M%S)_${DS}_${M}-$1-s$SEED"; mkdir -p runs/$ID
  echo "$M $DS split=$2 fixed=$3 seed=$SEED hp=$HP threads=$TH commit=$(git rev-parse --short HEAD)" > runs/$ID/cmd.txt
  (cd baselines/$M && DATA_PATH=../data/$DS/$2/ NUM_USER=$NU NUM_ITEM=$NI RUN_DIR=$ROOT/runs/$ID TF_THREADS=$TH NUMBA_NUM_THREADS=2 SEED=$SEED FIXED_EPOCHS=$3 \
     /mnt/data/zzx/conda_envs/tf115/bin/python -u $SCRIPT --dataset $OD --runid $ID $HP > $ROOT/runs/$ID/stdout.log 2>&1)
  echo $ID
}
if [ -z "$VAL" ]; then VAL=runs/$(run val fit 0); else echo "reuse $VAL"; fi
E=$(cat $VAL/best_epoch.txt) || { echo "FAILED: no best_epoch in $VAL"; exit 1; }
/mnt/data/zzx/conda_envs/odsr/bin/python baselines/eval_emb.py $DS $VAL fit 2>/dev/null | tail -1
echo "E*=$E (val run $VAL)"
FULL=runs/$(run full-E$E full $E)
/mnt/data/zzx/conda_envs/odsr/bin/python baselines/eval_emb.py $DS $FULL full 2>/dev/null | tail -1
echo "BASELINE_DONE $M $DS seed=$SEED E*=$E full=$FULL"
