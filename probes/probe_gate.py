"""Step-0 probe 3: does the best graph/item fusion weight depend on the user?
Per-user Recall@20 / NDCG@20 for a grid of fusion weights, then compare
(a) one global weight, (b) one weight per user-degree bucket (oracle on test = headroom upper bound),
(c) per-bucket weight chosen on the other half of users (2-fold, honest estimate)."""
import sys, os, json, torch, numpy as np, scipy.sparse as sp, importlib.util
off_src, data_root, ds, ckpt, B_path, out = sys.argv[1:7]
sys.path.insert(0, off_src)
os.environ["SELFREC_DATA_ROOT"] = os.path.abspath(data_root)
from common import GraphData
from ODSR import ODSR
torch.set_num_threads(16)
dev = torch.device("cpu")
data = GraphData(ds, dev, social_norm="row")
model = ODSR(data, dict(emb_size=64, n_layer=3, dropout=0.0, reg=0.0, eps=0.0, ssl_reg=0.0, tau=0.2), dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True); model.eval()
with torch.no_grad():
    fu, fi = model(None)
u, i = np.array(data.train_data).T
R = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(data.user_num, data.item_num)); R.data[:] = 1
P = torch.from_numpy(np.load(B_path)) if B_path.endswith(".npy") else None
if P is None:
    G = torch.from_numpy((R.T @ R).toarray()); G.diagonal().add_(float(B_path))
    P = torch.linalg.inv(G); del G; P.div_(-P.diagonal().clone()[None, :]); P.fill_diagonal_(0)
Rt = torch.from_numpy(R.toarray())
z = lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
W = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1e6]   # 1e6 ~ item model only
users = np.array(sorted(data.test_dict)); disc = 1.0 / np.log2(np.arange(2, 22))
rec = np.zeros((len(users), len(W)), np.float32); ndc = np.zeros_like(rec)
for st in range(0, len(users), 1024):
    b = users[st:st + 1024]; bt = torch.from_numpy(b)
    zg, ze = z(fu[bt] @ fi.T), z(Rt[bt] @ P)
    mask = torch.from_numpy(R[b].toarray() > 0)
    for k, w in enumerate(W):
        s = (zg + w * ze) if w < 1e5 else ze
        s = s.masked_fill(mask, -float("inf"))
        top = torch.topk(s, 20, dim=1).indices.numpy()
        for row, uid in enumerate(b):
            g = set(data.test_dict[uid]); hit = np.fromiter((t in g for t in top[row]), np.float32, 20)
            rec[st + row, k] = hit.sum() / len(g); ndc[st + row, k] = (hit * disc).sum() / disc[:min(20, len(g))].sum()
deg = np.asarray(R.sum(1)).ravel()[users]
sdeg = np.asarray((data.adj_uu.to_dense() > 0).sum(1)).ravel()[users] - 1
qs = np.quantile(deg, [0.2, 0.4, 0.6, 0.8]); bucket = np.searchsorted(qs, deg, side="right")
res = {"weights": W, "global": {}, "per_bucket": {}}
gbest = int(np.argmax(rec.mean(0)))
res["global"] = {"w": W[gbest], "recall@20": float(rec[:, gbest].mean()), "ndcg@20": float(ndc[:, gbest].mean())}
print(f"{ds} global best w={W[gbest]}: R@20={rec[:, gbest].mean():.4f} N@20={ndc[:, gbest].mean():.4f}")
oracle = np.zeros(len(users)); kb = {}
for bk in range(5):
    m = bucket == bk; k = int(np.argmax(rec[m].mean(0))); kb[bk] = k; oracle[m] = rec[m, k]
    res["per_bucket"][bk] = {"deg_range": [float(deg[m].min()), float(deg[m].max())], "n": int(m.sum()),
                             "mean_social_deg": float(sdeg[m].mean()), "best_w": W[k],
                             "recall@20_by_w": [float(x) for x in rec[m].mean(0)]}
    print(f"  bucket {bk} deg {deg[m].min():.0f}-{deg[m].max():.0f} n={m.sum()} soc={sdeg[m].mean():.1f} best w={W[k]} | R@20 by w: " +
          " ".join(f"{x:.4f}" for x in rec[m].mean(0)))
print(f"  oracle per-bucket R@20={oracle.mean():.4f} (headroom vs global {oracle.mean()/rec[:, gbest].mean()-1:+.2%})")
# honest 2-fold: choose per-bucket weight on one half of users, evaluate on the other
rng = np.random.default_rng(0); half = rng.random(len(users)) < 0.5; hr, hn = np.zeros(len(users)), np.zeros(len(users))
for fold in (half, ~half):
    for bk in range(5):
        msel, mev = (bucket == bk) & ~fold, (bucket == bk) & fold
        k = int(np.argmax(rec[msel].mean(0))); hr[mev] = rec[mev, k]; hn[mev] = ndc[mev, k]
print(f"  2-fold per-bucket R@20={hr.mean():.4f} N@20={hn.mean():.4f} (vs global {hr.mean()/rec[:, gbest].mean()-1:+.2%})")
res["oracle_bucket_recall@20"] = float(oracle.mean()); res["twofold_bucket"] = {"recall@20": float(hr.mean()), "ndcg@20": float(hn.mean())}
json.dump(res, open(out, "w"), indent=1)
print("DONE")
