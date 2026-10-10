"""Late fusion of the social graph view (trained ODSR) and the closed-form item-item view.

  select: python -m odsrx.fuse --dataset epinions --val_ratio 0.2 --graph runs/<val-run>/best.pth ...
          -> picks (prior hyper-parameters, fusion weight w) on the validation split
  apply : python -m odsrx.fuse --dataset epinions --graph runs/<full-run>/best.pth --w 0.3 ...
          -> refits the prior on full train and evaluates once on test

score_u = z_u(graph) + w * z_u(item prior); z_u = per-user standardisation over the catalogue.
"""
import argparse
import itertools
import json
import os

import torch

from .data import SocialRecData
from .evaluate import evaluate
from .model import HybridODSR
from .prior import item_prior_scores


class Fused(torch.nn.Module):
    def __init__(self, graph: HybridODSR, prior: torch.Tensor, w: float):
        super().__init__()
        self.graph, self.prior, self.w = graph, prior, w

    def context(self):
        return self.graph.context()

    @torch.no_grad()
    def score_users(self, users, ctx):
        g = self.graph.score_users(users, ctx)
        g = (g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-8)
        return g + self.w * self.prior[users]


def load_graph(data, path):
    m = HybridODSR(data, None)
    sd = torch.load(path, map_location="cpu")
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    m.backbone.load_state_dict(sd, strict=True)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(os.path.dirname(__file__), "..", "dataset"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.0)
    ap.add_argument("--lams", default="100")
    ap.add_argument("--ps", default="0")
    ap.add_argument("--betas", default="0")
    ap.add_argument("--ws", default="0,0.1,0.25,0.5,1,2,4")
    ap.add_argument("--w", type=float, default=None, help="apply mode: fixed weight")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cpu = torch.device("cpu")
    data = SocialRecData(args.data_root, args.dataset, cpu, val_ratio=args.val_ratio)
    print(data.summary(), flush=True)
    graph = load_graph(data, args.graph)
    target = data.eval_dict  # validation in select mode, test in apply mode
    cache = os.path.join(os.path.dirname(__file__), "..", "cache")
    ws = [args.w] if args.w is not None else [float(x) for x in args.ws.split(",")]
    res = {"args": vars(args), "split": "val" if args.val_ratio > 0 else "test", "grid": []}
    res["graph_only"] = evaluate(graph, data, target)
    print("graph only:", res["graph_only"], flush=True)
    for lam, p, beta in itertools.product(*[[float(x) for x in a.split(",")] for a in (args.lams, args.ps, args.betas)]):
        prior = item_prior_scores(data.R, lam, p, beta, cache_dir=cache)
        for w in ws:
            m = evaluate(Fused(graph, prior, w), data, target)
            res["grid"].append({"lam": lam, "p": p, "beta": beta, "w": w, **m})
            print(f"lam={lam} p={p} beta={beta} w={w}: R@20={m['recall@20']:.4f} N@20={m['ndcg@20']:.4f}", flush=True)
        del prior
    res["best"] = max(res["grid"], key=lambda r: r["recall@20"])
    print("BEST", res["best"], flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
