# ODSR-X: closed-form item and social views for robust social recommendation

ODSR-X adds two training-free views to the ODSR graph model (KDD'26, [official code](https://github.com/Rongfeng-Guo/ODSR-KDD-2026)) and fuses them late:

```
score(u, i) = z_u(ODSR graph) + w * z_u(item view) + w_soc * z_u(social view)
```

- **Graph view**: the ODSR backbone, unchanged (`odsrx/backbone.py`; state-dict compatible with the official checkpoints).
- **Item view**: an EASE-family closed-form item-item model with dropout regularisation and degree normalisation (`odsrx/prior.py`).
- **Social view**: the mean of a user's friends' interaction rows (`odsrx/prior.social_scores`).
- `z_u` standardises scores per user over the whole catalogue.

The fusion weights and the item-view hyper-parameters are chosen on a validation split carved out of the training data. The validation split takes 20% of each user's training items, for users with at least 5 items (seed 0).

## Results (Recall@20 / NDCG@20, full-catalogue ranking)

**Our protocol.** Everything is chosen on validation only: the weights, the item-view hyper-parameters and the graph's training epochs. The graph is then retrained on the full training set for the selected number of epochs, and the test set is evaluated **once**. We report the mean over 3 seeds (2026–2028).

| Method | Douban-Book | Epinions | Yelp2018 |
|---|---|---|---|
| GBSR, KDD'24 (paper) | .1694 / .1523 | .0793 / .0464 | — |
| SGIL, SIGIR'25 (paper) | .1809 / .1627 | .0844 / .0489 | .0797 / .0524 † |
| ODSR, KDD'26 (paper) | .1867 / .1669 | .0885 / .0509 | .0817 / .0536 |
| ODSR, same code and our protocol | .1822 / .1634 | .0824 / .0476 | .0797 / .0528 |
| ODSR, learning rate tuned on validation | (no gain) | .0863 / .0498 | in progress |
| **ODSR-X** | **.2180 / .2043** | **.0940 / .0546** | **.0828 / .0548** |

† SGIL on Yelp2018 as reported in the ODSR paper.

**Paper numbers** use the ODSR / GBSR / SGIL protocol: the test set is evaluated every epoch and the best epoch is kept. Same-protocol reruns of GBSR and SGIL are in `baselines/` and are still running.

### Caveats

- **Douban-Book**: almost all of the gain comes from the item view. The item view alone reaches .2172 / .2043. The social view's weight is selected as 0 on validation.
- **Epinions**: the gain is shared between views. The item view adds about +8.5% over the same-code ODSR, and the social view adds a further +5.1%.
- **Yelp2018**: the margin over the ODSR paper is small (+1.3% / +2.2%), although every seed is above it.
- **Official checkpoints**: the official ODSR checkpoints score exactly the paper numbers. However, retraining with the released `main.py` and config reaches only about .083 on Epinions, so the checkpoints are not reproducible from the released code. Our implementation follows the official training curve epoch by epoch (`progress.md`).

## Layout

| Path | Contents |
|---|---|
| `odsrx/` | data loading (same id mapping and graphs as ODSR), backbone, training, evaluation, closed-form views, fusion |
| `scripts/` | run launchers and the exact chains used for every reported number |
| `probes/` | step-0 diagnostics (oracle / headroom probes) |
| `baselines/` | same-protocol GBSR / SGIL: data export, protocol patch script, chain runner, embedding evaluator |
| `tests/` | equivalence tests against the official ODSR code and checkpoints |
| `source/`, `experiments/`, `docs/UPSTREAM_README.md` | the upstream tuned-ODSR code this project started from (kept unchanged) |
| `progress.md` | full research log: every run, number, failure and decision (in Chinese) |

## Usage

```bash
# data: dataset/{douban-book,epinions,yelp2018}/{train,test,trust}.txt
#   Douban-Book: SELFRec files; Epinions: GBSR public .npy exported as "u i" lines (probes/conv_gbsr_npy.py);
#   Yelp2018: the official ODSR Git LFS files. SHA-256 values are in dataset_checksums.json.

# 1) graph on the fit split, epoch chosen on validation
python -m odsrx.train --dataset epinions --prior none --val_ratio 0.2 --run_dir runs/epi_val
# 2) choose item-view hyper-parameters and fusion weights on validation
python -m odsrx.fuse --dataset epinions --val_ratio 0.2 --graph runs/epi_val/best.pth \
    --lams 200 --ps 0.25 --betas 0.25 --ws 0.1,0.25,0.5 --soc_ws 0,0.05,0.1,0.15,0.2,0.3 --out select.json
# 3) retrain the graph on the full training set for E* epochs, then evaluate the test set once
python -m odsrx.train --dataset epinions --prior none --max_epoch 4 --eval_interval 4 --run_dir runs/epi_full
python -m odsrx.fuse --dataset epinions --graph runs/epi_full/best.pth --lams 200 --ps 0.25 --betas 0.25 \
    --w 0.25 --soc_ws 0.2 --out apply.json
```

**Baselines.** `python baselines/apply_patches.py` clones the official GBSR and SGIL repositories at the commits we used and writes protocol-patched copies. The official repositories have no license, so their code is not redistributed here. Each change is marked `# PATCH`, and the models, losses and hyper-parameters are left untouched. Then run `bash baselines/run_baseline.sh <gbsr|sgil> <dataset> <threads> <seed>`.

## License

The MIT license (see `LICENSE`) covers our code in `odsrx/`, `scripts/`, `probes/`, `baselines/*.py|*.sh` and `tests/`. `odsrx/backbone.py` is adapted from the MIT-licensed official ODSR code (© 2026 Rongfeng-Guo). Files under `source/` and `experiments/` come from the upstream repository and keep its terms.
