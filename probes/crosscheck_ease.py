"""Score EASE through ODSR's own GraphData + evaluate_full_ranking (no re-implemented metric)."""
import sys, os, torch, numpy as np, scipy.sparse as sp
src, data_root, ds, lam = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
sys.path.insert(0, src)
os.environ["SELFREC_DATA_ROOT"] = os.path.abspath(data_root)
from common import GraphData
from utils import evaluate_full_ranking
torch.set_num_threads(16)
dev = torch.device("cpu")
data = GraphData(ds, dev, social_norm="row")
u, i = np.array(data.train_data).T
R = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(data.user_num, data.item_num)); R.data[:] = 1
G = torch.from_numpy((R.T @ R).toarray()); G.diagonal().add_(lam)
P = torch.linalg.inv(G); P.div_(-P.diagonal().clone()[None, :]); P.fill_diagonal_(0)
Rt = torch.from_numpy(R.toarray())
class EASEModel(torch.nn.Module):
    def encode(self, current_ssl_mode=None, return_context=False): return None, None, {}
    def score_all_items(self, users, context, item_batch=512): return Rt[users] @ P
print(ds, f"lam={lam}", evaluate_full_ranking(EASEModel(), data, dev, k_list=(10, 20), eval_batch=256), flush=True)
