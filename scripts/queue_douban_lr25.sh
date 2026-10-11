#!/usr/bin/env bash
# Start the Douban lr 2.5e-4 validation run once the Yelp social chain has released its CPU/memory.
cd /mnt/data/zzx/Code/ODSR
until grep -qE "CHAIN_SOC_DONE|FAILED" logs/chain_social_yelp2018.log; do sleep 120; done
scripts/launch.sh odsr-val-lr2.5e-4-s2026 douban-book --device cpu --threads 6 --seed 2026 --prior none --lr 2.5e-4 --val_ratio 0.2 --max_epoch 80 --eval_interval 1 --patience 8
