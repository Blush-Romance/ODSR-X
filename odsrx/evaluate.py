"""Full-catalogue ranking evaluation; same formulas as upstream source/utils.py
(training items masked; Recall@K = hits / |test set|; NDCG@K with ideal DCG over min(K, |test set|))."""
from typing import Dict, Iterable

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, data, eval_dict: Dict[int, list], ks: Iterable[int] = (10, 20), batch: int = 1024) -> Dict[str, float]:
    model.eval()
    ctx = model.context()
    ks = tuple(ks)
    kmax = max(ks)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    users = np.array(sorted(eval_dict))
    rec = {k: [] for k in ks}
    ndcg = {k: [] for k in ks}
    R = data.R
    for st in range(0, len(users), batch):
        b = users[st:st + batch]
        s = model.score_users(torch.from_numpy(b).to(data.device), ctx).float()
        rows, cols = R[b].nonzero()
        s[torch.from_numpy(rows).to(s.device), torch.from_numpy(cols).to(s.device)] = -float("inf")
        top = torch.topk(s, kmax, dim=1).indices.cpu().numpy()
        for r, u in enumerate(b):
            gt = set(eval_dict[u])
            hit = np.fromiter((t in gt for t in top[r]), np.float64, kmax)
            for k in ks:
                rec[k].append(hit[:k].sum() / len(gt))
                ndcg[k].append((hit[:k] * disc[:k]).sum() / disc[:min(k, len(gt))].sum())
    model.train()
    out = {f"recall@{k}": float(np.mean(rec[k])) for k in ks}
    out.update({f"ndcg@{k}": float(np.mean(ndcg[k])) for k in ks})
    return out
