"""Equivalence checks for the refactor (run from the project root)."""
import os, sys, torch, numpy as np
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "source"))
os.environ["SELFREC_DATA_ROOT"] = os.path.join(ROOT, "dataset")
from common import GraphData
from odsrx.data import SocialRecData
from odsrx.model import HybridODSR
from odsrx.evaluate import evaluate
from odsrx.prior import item_prior_scores
torch.set_num_threads(16)
ds, ckpt = sys.argv[1], sys.argv[2]
cpu = torch.device("cpu")
old = GraphData(ds, cpu, social_norm="row", verbose=False)
new = SocialRecData(os.path.join(ROOT, "dataset"), ds, cpu)
assert (old.user_num, old.item_num) == (new.user_num, new.item_num)
assert {u: sorted(v) for u, v in old.test_dict.items()} == {u: sorted(v) for u, v in new.test_dict.items()}
for name in ["adj_ui", "adj_uu", "norm_adj"]:
    a, b = getattr(old, name).to_dense(), getattr(new, name).to_dense()
    print(f"{name}: max|diff|={(a - b).abs().max().item():.3e}")
    assert torch.allclose(a, b, atol=1e-6)
m = HybridODSR(new, None)
m.backbone.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
print("ODSR official ckpt via odsrx:", evaluate(m, new, new.test_dict))
prior = item_prior_scores(new.R, 100.0, 0.0, 0.0, cache_dir=os.path.join(ROOT, "cache"))
h = HybridODSR(new, prior, alpha_init=1e4, learn_alpha=False)
h.backbone.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
print("alpha=1e4 (~EASE only):", evaluate(h, new, new.test_dict))
h2 = HybridODSR(new, prior, alpha_init=1.0)
u = torch.randint(0, new.user_num, (256,)); loss, parts = h2.compute_loss(u, torch.randint(0, new.item_num, (256,)), torch.randint(0, new.item_num, (256,)))
loss.backward(); print("backward ok", parts, "alpha grad", h2.log_alpha.grad.item())
print("ALL CHECKS PASSED")
