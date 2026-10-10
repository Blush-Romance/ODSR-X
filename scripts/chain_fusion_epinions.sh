#!/usr/bin/env bash
cd /mnt/data/zzx/Code/ODSR
until grep -qE "DONE|Traceback|Killed" logs/step0/linear_epinions.log; do sleep 30; done
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
B=$(ls results/step0/linear_epinions_EASE_B_lam*.npy | head -1)
REPO_UTILS=source/utils.py python -I probes/probe_fusion.py ../ODSR_official dataset epinions ../ODSR_official/checkpoints/Epinions.pth "$B" results/step0/fusion_epinions.json
echo CHAIN_DONE
