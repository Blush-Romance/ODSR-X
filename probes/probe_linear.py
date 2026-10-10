"""Step-0 probe: training-free linear recommenders under the exact ODSR evaluation protocol."""
import sys, os, re, json, math, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds

root, ds, out = sys.argv[1], sys.argv[2], sys.argv[3]

def pairs(p):
    with open(p) as f:
        return [tuple(re.split(r"[\s,]+", l.strip())[:2]) for l in f if l.strip()]

tr, te, so = pairs(f"{root}/{ds}/train.txt"), pairs(f"{root}/{ds}/test.txt"), pairs(f"{root}/{ds}/trust.txt")
um, im = {}, {}
gid = lambda m, k: m.setdefault(k, len(m))
trn = [(gid(um, u), gid(im, i)) for u, i in tr]
test = {}
for u, i in te:
    if u in um and i in im:
        test.setdefault(um[u], []).append(im[i])
U, I = len(um), len(im)
r, c = np.array(trn).T
R = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(U, I))
R.data[:] = 1.0
S = set()
for u, v in so:
    if u in um and v in um and um[u] != um[v]:
        a, b = sorted((um[u], um[v])); S.add((a, b))
S = np.array(list(S)).T
A = sp.csr_matrix((np.ones(2 * S.shape[1], np.float32), (np.r_[S[0], S[1]], np.r_[S[1], S[0]])), shape=(U, U))
users = np.array(sorted(test))
print(f"{ds}: U={U} I={I} train={R.nnz} test_users={len(users)} social={A.nnz//2}", flush=True)

def evaluate(score_fn, bs=1024):
    rec = {10: [], 20: []}; nd = {10: [], 20: []}
    disc = 1.0 / np.log2(np.arange(2, 22))
    for st in range(0, len(users), bs):
        b = users[st:st + bs]
        s = np.asarray(score_fn(b), dtype=np.float32)
        s[R[b].nonzero()] = -np.inf
        top = np.argpartition(-s, 20, axis=1)[:, :20]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), 1), 1)
        for row, u in enumerate(b):
            g = set(test[u]); hit = np.array([t in g for t in top[row]], dtype=np.float32)
            for k in (10, 20):
                rec[k].append(hit[:k].sum() / len(g))
                nd[k].append((hit[:k] * disc[:k]).sum() / disc[:min(k, len(g))].sum())
    return {f"recall@{k}": float(np.mean(rec[k])) for k in (10, 20)} | {f"ndcg@{k}": float(np.mean(nd[k])) for k in (10, 20)}

res = {}
def log(name, m):
    res[name] = m; print(f"{name:32s} R@20={m['recall@20']:.4f} N@20={m['ndcg@20']:.4f} R@10={m['recall@10']:.4f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)

# Popularity
pop = np.asarray(R.sum(0)).ravel()
log("Pop", evaluate(lambda b: np.tile(pop, (len(b), 1))))

# GF-CF (Shen et al., CIKM 2021)
du = np.asarray(R.sum(1)).ravel(); di = np.asarray(R.sum(0)).ravel()
du_is = np.power(du, -0.5, where=du > 0); di_is = np.power(di, -0.5, where=di > 0); di_s = np.sqrt(di)
Rn = sp.diags(du_is) @ R @ sp.diags(di_is)
P = (Rn.T @ Rn).toarray().astype(np.float32)
t = time.time(); _, _, Vt = svds(Rn.astype(np.float64), k=256); Vt = Vt.astype(np.float32); print(f"svd {time.time()-t:.0f}s", flush=True)
for a in (0.0, 0.3, 1.0):
    log(f"GF-CF a={a}", evaluate(lambda b: R[b] @ P + a * ((R[b] @ sp.diags(di_is)) @ Vt.T @ Vt) * di_s[None, :]))
del P

# EASE (Steck, WWW 2019); lambda grid reported in full
import torch
torch.set_num_threads(16)
G = torch.from_numpy((R.T @ R).toarray().astype(np.float32))
diag = np.diag_indices(I)
best = None
lams = [float(x) for x in os.environ.get("EASE_LAMS", "50,100,200,400,800,1600").split(",")]
for lam in lams:
    t = time.time(); G.diagonal().add_(lam); Pm = torch.linalg.inv(G); G.diagonal().sub_(lam)
    Pm.div_(-Pm.diagonal().clone()[None, :]); Pm.fill_diagonal_(0); B = Pm.numpy()
    m = evaluate(lambda b: R[b] @ B); log(f"EASE lam={lam}", m); print(f"  inv {time.time()-t:.0f}s", flush=True)
    if best is None or m["recall@20"] > best[0]: best = (m["recall@20"], lam, B.copy())
    del Pm, B
_, lam, B = best
np.save(out.replace(".json", f"_EASE_B_lam{lam}.npy"), B)

# Social augmentation of the user row: r_u + beta * mean over friends
Arow = sp.diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
for beta in (0.1, 0.3, 1.0):
    log(f"EASE lam={lam}+social b={beta}", evaluate(lambda b: (R[b] + beta * (Arow[b] @ R)) @ B))
print("DONE", flush=True)
