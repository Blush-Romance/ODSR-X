#!/usr/bin/env bash
# The SGIL-Epinions validation run was started as runs/smoke_sgil_epi (fit split, official seed 2024); reuse it for E*.
cd /mnt/data/zzx/Code/ODSR
until grep -q "Evaluation results" runs/smoke_sgil_epi/training.txt 2>/dev/null; do sleep 300; done
bash baselines/run_baseline.sh sgil epinions 6 2026 runs/smoke_sgil_epi
