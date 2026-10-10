"""Step-0 probe 2: closed-form item-item variants + closed-form social denoising.
Hyper-parameters are selected on a validation split carved from train (20% per user, seed 0),
then refit on the full train set and evaluated once on test. Test numbers for every config are
also logged for transparency, but selection never looks at them."""
import sys, os, re, json, time, itertools
import numpy as np, scipy.sparse as sp, torch

root, ds, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
rng = np.random.default_rng(0)

def pairs(p):
    with open(p) as f:
        return [tuple(re.split(r"[\s,]+", l.strip())[:2]) for l in f if l.strip()]

tr, te, so = pairs(f"{root}/{ds}/train.txt"), pairs(f"{root}/{ds}/test.txt"), pairs(f"{root}/{ds}/trust.txt")
um, im = {}, {}
gid = lambda m, k: m.setdefault(k, len(m))
trn = np.array([(gid(um, u), gid(im, i)) for u, i in tr])
U, I = len(um), len(im)
test = {}
for u, i in te:
    if u in um and i in im:
        test.setdefault(um[u], set()).add(im[i])
S = set()
for u, v in so:
    if u in um and v in um and um[u] != um[v]:
        S.add(tuple(sorted((um[u], um[v]))))
S = np.array(sorted(S)).T
A = sp.csr_matrix((np.ones(2 * S.shape[1], np.float32), (np.r_[S[0], S[1]], np.r_[S[1], S[0]])), shape=(U, U))

def csr(p):
    m = sp.csr_matrix((np.ones(len(p), np.float32), (p[:, 0], p[:, 1])), shape=(U, I)); m.data[:] = 1; return m

# validation split: 20% of each user's train items (users with >= 5 items)
perm = rng.permutation(len(trn)); trn_p = trn[perm]
order = np.argsort(trn_p[:, 0], kind="stable"); trn_p = trn_p[order]
_, start, cnt = np.unique(trn_p[:, 0], return_index=True, return_counts=True)
is_val = np.zeros(len(trn_p), bool)
for s0, c in zip(start, cnt):
    if c >= 5:
        is_val[s0:s0 + int(round(0.2 * c))] = True
R_full, R_fit = csr(trn), csr(trn_p[~is_val])
val = {}
for u, i in trn_p[is_val]:
    val.setdefault(int(u), set()).add(int(i))
print(f"{ds}: U={U} I={I} train={R_full.nnz} fit={R_fit.nnz} val_users={len(val)} test_users={len(test)} social={A.nnz//2}", flush=True)

disc = 1.0 / np.log2(np.arange(2, 22))
def evaluate(R_in, gt, score_fn, bs=2048):
    users = np.array(sorted(gt)); r20, n20 = [], []
    for st in range(0, len(users), bs):
        b = users[st:st + bs]
        s = score_fn(b)
        s = s.numpy() if torch.is_tensor(s) else np.asarray(s)
        s = s.astype(np.float32, copy=True); s[R_in[b].nonzero()] = -np.inf
        top = np.argpartition(-s, 20, axis=1)[:, :20]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), 1), 1)
        for row, u in enumerate(b):
            g = gt[u]; hit = np.fromiter((t in g for t in top[row]), np.float32, 20)
            r20.append(hit.sum() / len(g)); n20.append((hit * disc).sum() / disc[:min(20, len(g))].sum())
    return {"recall@20": float(np.mean(r20)), "ndcg@20": float(np.mean(n20))}

def item_model(R, lam, p, beta):
    """EASE family. p: dropout emphasis (Lambda = p/(1-p) diag(G) + lam I); beta: degree normalisation of the Gram."""
    G = torch.from_numpy((R.T @ R).toarray())
    d = G.diagonal().clone()
    if beta > 0:
        s = d.clamp(min=1).pow(-beta / 2); G.mul_(s[:, None]).mul_(s[None, :]); d = G.diagonal().clone()
    G.diagonal().add_(lam + p / (1 - p) * d)
    P = torch.linalg.inv(G); del G
    P.div_(-P.diagonal().clone()[None, :]); P.fill_diagonal_(0)
    if beta > 0:
        s = torch.from_numpy(np.asarray(R.sum(0)).ravel().astype(np.float32)).clamp(min=1).pow(beta / 2)
        P.mul_(1 / s[:, None]).mul_(s[None, :])  # map back to raw-input space
    return P

def social_weights(R, gamma, kmax=50):
    """Per-user ridge regression of the user's own row on friends' rows: learns which friends are reliable.
    Returns a sparse U x U matrix W (row u: weights over friends of u)."""
    Rd = R.tocsr(); rows, cols, vals = [], [], []
    deg_soc = np.diff(A.indptr)
    for u in range(U):
        f = A.indices[A.indptr[u]:A.indptr[u + 1]]
        if len(f) == 0 or Rd.indptr[u + 1] == Rd.indptr[u]:
            continue
        if len(f) > kmax:  # keep the friends with largest co-interaction overlap
            ov = np.asarray((Rd[f] @ Rd[u].T).todense()).ravel(); f = f[np.argsort(-ov)[:kmax]]
        Xf = Rd[f]; K = (Xf @ Xf.T).toarray(); b = np.asarray((Xf @ Rd[u].T).todense()).ravel()
        w = np.linalg.solve(K + gamma * np.eye(len(f)), b)
        rows += [u] * len(f); cols += list(f); vals += list(w)
    return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(U, U))

res = {}
def log(name, cfg, vm, tm):
    res[name] = {"cfg": cfg, "val": vm, "test": tm}
    print(f"{name:44s} VAL R@20={vm['recall@20']:.4f} N@20={vm['ndcg@20']:.4f} | TEST R@20={tm['recall@20']:.4f} N@20={tm['ndcg@20']:.4f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)

lams = [float(x) for x in os.environ.get("LAMS", "50,100,200,400").split(",")]
ps = [float(x) for x in os.environ.get("PS", "0,0.1,0.25,0.5").split(",")]
betas = [float(x) for x in os.environ.get("BETAS", "0,0.25,0.5").split(",")]

# Stage 1: item model grid on validation (fit split); test column uses the full-train refit only for the val-best config
best = None
for lam, p, beta in itertools.product(lams, ps, betas):
    t = time.time(); B = item_model(R_fit, lam, p, beta)
    Rf = torch.from_numpy(R_fit.toarray())
    vm = evaluate(R_fit, val, lambda b: Rf[b] @ B); del B, Rf
    print(f"  item lam={lam} p={p} beta={beta}: VAL R@20={vm['recall@20']:.4f} N@20={vm['ndcg@20']:.4f} ({time.time()-t:.0f}s)", flush=True)
    res.setdefault("_grid_item", []).append({"lam": lam, "p": p, "beta": beta, "val": vm})
    if best is None or vm["recall@20"] > best[0]:
        best = (vm["recall@20"], dict(lam=lam, p=p, beta=beta))
cfg_item = best[1]
B_fit = item_model(R_fit, **cfg_item); B_full = item_model(R_full, **cfg_item)
Rf, Rt = torch.from_numpy(R_fit.toarray()), torch.from_numpy(R_full.toarray())
vm = evaluate(R_fit, val, lambda b: Rf[b] @ B_fit)
tm = evaluate(R_full, test, lambda b: Rt[b] @ B_full)
log("ITEM(val-best)", cfg_item, vm, tm)
base = dict(lam=cfg_item["lam"], p=0.0, beta=0.0)
if base != cfg_item:
    Bb = item_model(R_full, **base); Bv = item_model(R_fit, **base)
    log("EASE(same lam, p=0, beta=0)", base, evaluate(R_fit, val, lambda b: Rf[b] @ Bv), evaluate(R_full, test, lambda b: Rt[b] @ Bb)); del Bb, Bv

# Stage 2: closed-form social denoising on top of the selected item model
Arow = sp.diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
def run_social(name, Wv, Wt, mode, coef):
    Wv_t = torch.from_numpy(Wv.toarray()) if sp.issparse(Wv) else Wv
    Wt_t = torch.from_numpy(Wt.toarray()) if sp.issparse(Wt) else Wt
    if mode == "input":   # augment the user's input row, then item model
        fv = lambda b: (Rf[b] + coef * (Wv_t[b] @ Rf)) @ B_fit
        ft = lambda b: (Rt[b] + coef * (Wt_t[b] @ Rt)) @ B_full
    else:                 # late: add friends' item-model scores
        fv = lambda b: Rf[b] @ B_fit + coef * (Wv_t[b] @ Rf) @ B_fit
        ft = lambda b: Rt[b] @ B_full + coef * (Wt_t[b] @ Rt) @ B_full
    return evaluate(R_fit, val, fv), evaluate(R_full, test, ft)

best_s = None
for gamma in [float(x) for x in os.environ.get("GAMMAS", "1,5,20").split(",")]:
    t = time.time(); Wv = social_weights(R_fit, gamma); Wt = social_weights(R_full, gamma)
    print(f"  social weights gamma={gamma}: nnz={Wt.nnz} ({time.time()-t:.0f}s) mean w={Wt.data.mean():.4f} frac>0.05={(Wt.data>0.05).mean():.3f}", flush=True)
    for coef in (0.1, 0.3, 1.0):
        vm, tm = run_social(f"ridge g={gamma} c={coef}", Wv, Wt, "input", coef)
        log(f"ITEM + ridge-social input g={gamma} c={coef}", {"gamma": gamma, "coef": coef}, vm, tm)
        if best_s is None or vm["recall@20"] > best_s[0]: best_s = (vm["recall@20"], gamma, coef)
for coef in (0.05, 0.1, 0.3):
    vm, tm = run_social("mean", Arow, Arow, "input", coef)
    log(f"ITEM + mean-friends input c={coef}", {"coef": coef}, vm, tm)
print("BEST social (val):", best_s, flush=True)
print("DONE", flush=True)
