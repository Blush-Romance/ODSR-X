"""Score saved user/item embeddings of a baseline run with odsrx.evaluate (same evaluator as ODSR-X).
usage: eval_emb.py <dataset> <run_dir> <fit|full>   (fit -> validation split, full -> test)"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from odsrx.data import SocialRecData
from odsrx.evaluate import evaluate

class Emb(torch.nn.Module):
    def __init__(self, u, i): super().__init__(); self.u, self.i = u, i
    def context(self): return None
    def score_users(self, users, ctx): return self.u[users] @ self.i.t()

ds, run, tag = sys.argv[1:4]
torch.set_num_threads(8)
data = SocialRecData(os.path.join(os.path.dirname(__file__), "..", "dataset"), ds, torch.device("cpu"), val_ratio=0.2 if tag == "fit" else 0.0)
u = torch.from_numpy(np.load(os.path.join(run, "user_emb.npy"))).float()
i = torch.from_numpy(np.load(os.path.join(run, "item_emb.npy"))).float()
assert u.shape[0] == data.user_num and i.shape[0] == data.item_num, (u.shape, i.shape)
m = evaluate(Emb(u, i), data, data.eval_dict)
json.dump({"dataset": ds, "run": run, "split": "val" if tag == "fit" else "test", **m}, open(os.path.join(run, "odsrx_eval.json"), "w"), indent=1)
print(json.dumps(m))
