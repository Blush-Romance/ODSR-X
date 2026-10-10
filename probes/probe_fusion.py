"""Step-0 probe: is a closed-form item-item model (EASE) complementary to ODSR?
Loads the official ODSR checkpoint, scores the full catalog on CPU, and fuses with EASE
after per-user z-normalisation. Evaluation = the repo's evaluate_full_ranking."""
import sys, os, json, torch, numpy as np, scipy.sparse as sp
off_src, data_root, ds, ckpt, lam, out = sys.argv[1:7]
sys.path.insert(0, off_src)
os.environ["SELFREC_DATA_ROOT"] = os.path.abspath(data_root)
import importlib.util
from common import GraphData
from ODSR import ODSR
_spec = importlib.util.spec_from_file_location("repo_utils", os.environ["REPO_UTILS"])
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
evaluate_full_ranking = _mod.evaluate_full_ranking
torch.set_num_threads(16)
dev = torch.device("cpu")
data = GraphData(ds, dev, social_norm="row")
cfg = dict(emb_size=64, n_layer=3, dropout=0.0, reg=0.0, eps=0.0, ssl_reg=0.0, tau=0.2)
model = ODSR(data, cfg, dev)
sd = torch.load(ckpt, map_location="cpu")
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
print("load:", model.load_state_dict(sd, strict=True))
model.eval()
with torch.no_grad():
    fu, fi = model(None)
u, i = np.array(data.train_data).T
R = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(data.user_num, data.item_num)); R.data[:] = 1
if lam.endswith(".npy"):  # precomputed EASE weights saved by probe_linear.py (same ID mapping)
    P = torch.from_numpy(np.load(lam)); lam = os.path.basename(lam)
else:
    G = torch.from_numpy((R.T @ R).toarray()); G.diagonal().add_(float(lam))
    P = torch.linalg.inv(G); del G
    P.div_(-P.diagonal().clone()[None, :]); P.fill_diagonal_(0)
Rt = torch.from_numpy(R.toarray())
def z(x): return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
class Fused(torch.nn.Module):
    def __init__(self, w): super().__init__(); self.w = w
    def encode(self, current_ssl_mode=None, return_context=False): return None, None, {}
    def score_all_items(self, users, context, item_batch=512):
        g = fu[users] @ fi.T
        if self.w is None: return g
        if self.w == "ease": return Rt[users] @ P
        return z(g) + self.w * z(Rt[users] @ P)
res = {}
for w in [None, "ease", 0.25, 0.5, 1.0, 2.0, 4.0]:
    m = evaluate_full_ranking(Fused(w), data, dev, k_list=(10, 20), eval_batch=512)
    name = {None: "ODSR(official ckpt)", "ease": f"EASE lam={lam}"}.get(w, f"z(ODSR)+{w}*z(EASE)")
    res[name] = m; print(f"{ds} {name:28s} R@20={m['recall@20']:.4f} N@20={m['ndcg@20']:.4f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
