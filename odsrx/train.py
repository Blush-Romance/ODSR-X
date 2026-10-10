"""Train / evaluate ODSR and ODSR-X.

  python -m odsrx.train --dataset douban-book --prior none          # ODSR, in-pipeline baseline
  python -m odsrx.train --dataset douban-book --prior ease --prior_lam 50 --prior_p 0.1 --prior_beta 0.25

Model selection follows the protocol of ODSR / GBSR / SGIL (evaluate every `eval_interval`
epochs on test, keep the best Recall@20) unless --val_ratio > 0, in which case the epoch is
selected on a validation split carved from train and test is only reported.
"""
import argparse
import json
import os
import random
import socket
import subprocess
import time

import numpy as np
import torch

from .data import SocialRecData
from .evaluate import evaluate
from .model import HybridODSR
from .prior import item_prior_scores

# official ODSR per-dataset settings (github.com/Rongfeng-Guo/ODSR-KDD-2026, main.py)
DATASET_CFG = {
    "douban-book": dict(lr=1e-3, reg=0.0, ssl_reg=0.58, eps=0.57, dropout=0.01, max_epoch=60),
    "epinions": dict(lr=1e-3, reg=0.0, ssl_reg=0.18, eps=0.46, dropout=0.14, max_epoch=30),
    "yelp2018": dict(lr=1e-3, reg=0.0, ssl_reg=0.43, eps=0.54, dropout=0.17, max_epoch=20),
}
DATASET_CFG["yelp-gbsr"] = dict(DATASET_CFG["yelp2018"])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(os.path.dirname(__file__), "..", "dataset"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prior", choices=["none", "ease"], default="ease")
    ap.add_argument("--prior_lam", type=float, default=100.0)
    ap.add_argument("--prior_p", type=float, default=0.0)
    ap.add_argument("--prior_beta", type=float, default=0.0)
    ap.add_argument("--alpha_init", type=float, default=1.0)
    ap.add_argument("--fix_alpha", action="store_true")
    for k in ["lr", "reg", "ssl_reg", "eps", "dropout", "max_epoch"]:
        ap.add_argument(f"--{k}", type=float if k != "max_epoch" else int, default=None)
    ap.add_argument("--emb_size", type=int, default=64)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--eval_interval", type=int, default=1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val_ratio", type=float, default=0.0)
    ap.add_argument("--hard_ratio", type=float, default=0.0, help="fraction of negatives drawn from the prior's top-K unseen items")
    ap.add_argument("--hard_k", type=int, default=100)
    ap.add_argument("--init_backbone", default=None, help="optional official ODSR checkpoint to warm-start the backbone")
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    cfg = dict(DATASET_CFG.get(args.dataset, DATASET_CFG["douban-book"]))
    for k in cfg:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.run_dir, exist_ok=True)

    data = SocialRecData(args.data_root, args.dataset, dev, val_ratio=args.val_ratio)
    print(data.summary(), flush=True)

    prior = None
    if args.prior == "ease":
        t = time.time()
        prior = item_prior_scores(data.R, args.prior_lam, args.prior_p, args.prior_beta,
                                  cache_dir=os.environ.get("ODSRX_CACHE"))
        prior = prior.to(dev) if dev.type == "cuda" and prior.numel() < 6e8 else prior
        print(f"prior ready ({time.time()-t:.0f}s) shape={tuple(prior.shape)}", flush=True)

    model = HybridODSR(data, prior, args.emb_size, args.n_layer, cfg["dropout"], cfg["reg"], cfg["ssl_reg"],
                       cfg["eps"], args.tau, args.alpha_init, learn_alpha=not args.fix_alpha).to(dev)
    if args.init_backbone:
        model.backbone.load_state_dict(torch.load(args.init_backbone, map_location="cpu"), strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    meta = dict(vars(args), **{f"cfg_{k}": v for k, v in cfg.items()}, host=socket.gethostname(),
                torch=torch.__version__, start=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        meta["git_commit"] = subprocess.check_output(["git", "-C", os.path.dirname(__file__), "rev-parse", "HEAD"],
                                                     text=True).strip()
        meta["git_dirty"] = bool(subprocess.check_output(["git", "-C", os.path.dirname(__file__), "status",
                                                          "--porcelain", "odsrx"], text=True).strip())
    except Exception:
        pass
    json.dump(meta, open(os.path.join(args.run_dir, "meta.json"), "w"), indent=1)

    hard_pool = None
    if args.hard_ratio > 0:
        assert prior is not None, "prior-guided negatives need a prior"
        t = time.time(); hard_pool = torch.empty(data.user_num, args.hard_k, dtype=torch.long)
        for st in range(0, data.user_num, 2048):
            s = prior[st:st + 2048].float().cpu().clone()
            rows, cols = data.R[st:st + 2048].nonzero()
            s[torch.from_numpy(rows), torch.from_numpy(cols)] = -float("inf")
            hard_pool[st:st + 2048] = torch.topk(s, args.hard_k, dim=1).indices
        print(f"hard pool ready ({time.time()-t:.0f}s) k={args.hard_k} ratio={args.hard_ratio}", flush=True)
    users = torch.from_numpy(data.train_data[:, 0])
    items = torch.from_numpy(data.train_data[:, 1])
    n_batch = max(1, len(users) // args.batch_size)
    history, best, bad = [], None, 0
    if args.init_backbone:
        m0 = evaluate(model, data, data.eval_dict)
        print(f"init eval r20={m0['recall@20']:.6f} n20={m0['ndcg@20']:.6f}", flush=True)
    for ep in range(1, cfg["max_epoch"] + 1):
        t0 = time.time()
        perm = torch.from_numpy(np.random.permutation(len(users)))
        tot = {"loss": 0.0, "rec": 0.0, "ssl": 0.0}
        for bi in range(n_batch):
            b = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            u, p = users[b].to(dev), items[b].to(dev)
            n = torch.randint(0, data.item_num, (len(b),))
            if hard_pool is not None:
                use = torch.rand(len(b)) < args.hard_ratio
                hn = hard_pool[users[b], torch.randint(0, args.hard_k, (len(b),))]
                n = torch.where(use, hn, n)
            n = n.to(dev)
            loss, parts = model.compute_loss(u, p, n)
            opt.zero_grad(); loss.backward(); opt.step()
            tot["loss"] += loss.item(); tot["rec"] += parts["rec"]; tot["ssl"] += parts["ssl"]
        rec = {k: v / n_batch for k, v in tot.items()}
        line = f"epoch {ep} loss={rec['loss']:.5f} rec={rec['rec']:.5f} ssl={rec['ssl']:.5f} alpha={model.alpha.item():.4f} time={time.time()-t0:.0f}s"
        entry = {"epoch": ep, **rec, "alpha": model.alpha.item()}
        if ep % args.eval_interval == 0:
            m = evaluate(model, data, data.eval_dict)
            entry["eval"] = m
            if args.val_ratio > 0:
                entry["test"] = evaluate(model, data, data.test_dict)
            line += f" | eval r20={m['recall@20']:.6f} n20={m['ndcg@20']:.6f}"
            if "test" in entry:
                line += f" test r20={entry['test']['recall@20']:.6f} n20={entry['test']['ndcg@20']:.6f}"
            if best is None or m["recall@20"] > best["eval"]["recall@20"]:
                best, bad = entry, 0
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                           os.path.join(args.run_dir, "best.pth"))
            else:
                bad += 1
        print(line, flush=True)
        history.append(entry)
        json.dump({"meta": meta, "best": best, "history": history},
                  open(os.path.join(args.run_dir, "result.json"), "w"), indent=1)
        if bad >= args.patience:
            print(f"early stop at epoch {ep}", flush=True)
            break
    print("BEST", json.dumps(best), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
