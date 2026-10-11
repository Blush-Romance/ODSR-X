#!/usr/bin/env bash
# Wait for the Yelp seed chain (memory: dense 46k x 46k views), then run the Yelp social-view chain.
cd /mnt/data/zzx/Code/ODSR
until grep -q SEEDS_DONE logs/chain_seeds_yelp2018.log; do sleep 120; done
G() { echo $(ls -d runs/*yelp2018_odsr-full-E5-s$1 | head -1)/best.pth; }
bash scripts/chain_social.sh yelp2018 runs/hs-Ubuntu_20261011_013921_yelp2018_odsr-valsplit-s2026/best.pth 400 0.25 0 0.1,0.25,0.5 0,0.05,0.1,0.2,0.3 12 own_s2026=$(G 2026) own_s2027=$(G 2027) own_s2028=$(G 2028) official=../ODSR_official/checkpoints/Yelp2018.pth
