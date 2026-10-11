"""Step-0 headroom probe on the validation split: which extra closed-form views add to graph+EASE?
Views (all z-scored per user): G graph, E EASE(selected), RP3 (RP3beta), SOC (friends interactions),
SOCE (friends interactions through EASE), POP. Greedy forward selection of weights on val R@20."""
import argparse, json, sys, time
import numpy as np, scipy.sparse as sp, torch
from odsrx.data import SocialRecData
from odsrx.evaluate import evaluate
from odsrx.fuse import load_graph
from odsrx.prior import fit_item_prior

ap = argparse.ArgumentParser()
ap.add_argument("--dataset"); ap.add_argument("--graph"); ap.add_argument("--lam", type=float)
ap.add_argument("--p", type=float); ap.add_argument("--beta", type=float); ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--out")
a = ap.parse_args()
torch.set_num_threads(a.threads)
data = SocialRecData("dataset", a.dataset, torch.device("cpu"), val_ratio=0.2)
print(data.summary(), flush=True)
R = data.R.astype(np.float32).tocsr()
U, I = R.shape
z = lambda S: S.sub_(S.mean(1, keepdim=True)).div_(S.std(1, keepdim=True) + 1e-8)

def dense_scores(fn):
    S = torch.empty((U, I), dtype=torch.float32)
    for st in range(0, U, 4096):
        S[st:st + 4096] = fn(st, min(U, st + 4096))
    return z(S).half()

views = {}
g = load_graph(data, a.graph)
with torch.no_grad():
    fu, fi = g.context()
views["G"] = z(fu @ fi.t()).half()
t = time.time()
B = fit_item_prior(R, a.lam, a.p, a.beta)
X = torch.from_numpy(R.toarray())
views["E"] = dense_scores(lambda s, e: X[s:e] @ B)
print(f"EASE {time.time()-t:.0f}s", flush=True)
# social row-normalised (no self loop) taken from adj_uu structure
Ssoc = data.adj_uu.coalesce()
idx, val = Ssoc.indices().numpy(), Ssoc.values().numpy()
S = sp.csr_matrix((np.ones_like(val), (idx[0], idx[1])), shape=(U, U))
S.setdiag(0); S.eliminate_zeros()
deg = np.asarray(S.sum(1)).ravel(); deg[deg == 0] = 1
S = sp.diags(1 / deg) @ S
SR = (S @ R).tocsr()
views["SOC"] = dense_scores(lambda s, e: torch.from_numpy(SR[s:e].toarray()))
views["SOCE"] = dense_scores(lambda s, e: torch.from_numpy(SR[s:e].toarray()) @ B)
del B
torch.cuda.empty_cache() if torch.cuda.is_available() else None
# RP3beta (Paudel et al. 2016), alpha=1
pop = np.asarray(R.sum(0)).ravel(); pop[pop == 0] = 1
du = np.asarray(R.sum(1)).ravel(); du[du == 0] = 1
Pui = sp.diags(1 / du) @ R
Piu = sp.diags(1 / pop) @ R.T.tocsr()
for b in (0.5,):
    W = torch.from_numpy((Piu @ Pui).toarray().astype(np.float32)) * torch.from_numpy(pop ** -b).float()[None, :]
    W.fill_diagonal_(0)
    views[f"RP3"] = dense_scores(lambda s, e: X[s:e] @ W)
    del W
views["POP"] = z(torch.from_numpy(np.log1p(pop)).float()[None, :].repeat(U, 1)).half()
print("views:", list(views), flush=True)

class Mix(torch.nn.Module):
    def __init__(self, w): super().__init__(); self.w = w
    def context(self): return None
    def score_users(self, users, ctx):
        s = 0
        for k, w in self.w.items():
            if w: s = s + w * views[k][users].float()
        return s
ev = lambda w: evaluate(Mix(w), data, data.eval_dict)
res = {"single": {}, "steps": []}
for k in views:
    m = ev({k: 1.0}); res["single"][k] = m
    print(f"single {k}: R@20={m['recall@20']:.4f} N@20={m['ndcg@20']:.4f}", flush=True)
cur = {"G": 1.0}; best = ev(cur)["recall@20"]
grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
for rnd in range(4):
    cand = None
    for k in views:
        if k == "G": continue
        for w in grid:
            trial = dict(cur); trial[k] = w
            r = ev(trial)["recall@20"]
            if r > best + 1e-5 and (cand is None or r > cand[0]): cand = (r, k, w)
    if cand is None: break
    best = cand[0]; cur[cand[1]] = cand[2]
    m = ev(cur); res["steps"].append({"w": dict(cur), **m})
    print(f"step {rnd}: add {cand[1]}={cand[2]} -> R@20={m['recall@20']:.4f} N@20={m['ndcg@20']:.4f} w={cur}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("PROBE_DONE", flush=True)
