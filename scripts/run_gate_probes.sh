#!/usr/bin/env bash
cd /mnt/data/zzx/Code/ODSR
source /mnt/data/zzx/anaconda3/etc/profile.d/conda.sh; conda activate /mnt/data/zzx/conda_envs/odsr
python -I probes/probe_gate.py ../ODSR_official dataset epinions ../ODSR_official/checkpoints/Epinions.pth results/step0/linear_epinions_EASE_B_lam200.0.npy results/step0/gate_epinions.json
python -I probes/probe_gate.py ../ODSR_official dataset douban-book ../ODSR_official/checkpoints/Douban-Book.pth 100 results/step0/gate_douban.json
echo ALL_DONE
