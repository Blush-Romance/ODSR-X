"""Export our splits to the GBSR/SGIL .npy dict format, so both baselines see exactly the data ODSR-X sees.

  <out>/<dataset>/fit/   traindata.npy (train minus val), valdata.npy, testdata.npy, user_users_d.npy
  <out>/<dataset>/full/  traindata.npy (full train),      valdata.npy (= test, unused), testdata.npy, user_users_d.npy
Written as .json (dict u -> list); baselines/json_to_npy.py (run in the TF1/numpy-1.x env) turns them into the
pickled .npy dicts the official loaders expect (numpy-2 pickles do not load under numpy 1.x).
Ids are the internal ids of odsrx.data.SocialRecData (same mapping as ODSR). meta.json records sizes.
"""
import json, os, sys
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from odsrx.data import SocialRecData

def rows(R):
    d = defaultdict(list)
    for u in range(R.shape[0]):
        items = R.indices[R.indptr[u]:R.indptr[u + 1]].tolist()
        if items:
            d[u] = items
    return d

def social(data):
    idx = data.adj_uu.coalesce().indices().numpy()
    d = defaultdict(list)
    for u, v in zip(idx[0].tolist(), idx[1].tolist()):
        if u != v:
            d[u].append(v)
    return d

def dump(path, d):
    json.dump({str(k): v for k, v in d.items()}, open(path, "w"))

ds, out = sys.argv[1], sys.argv[2]
root = os.path.join(os.path.dirname(__file__), "..", "dataset")
meta = {}
for tag, vr in (("fit", 0.2), ("full", 0.0)):
    data = SocialRecData(root, ds, torch.device("cpu"), val_ratio=vr)
    d = os.path.join(out, ds, tag); os.makedirs(d, exist_ok=True)
    test = defaultdict(list, {int(u): list(map(int, v)) for u, v in data.test_dict.items()})
    val = defaultdict(list, {int(u): list(map(int, v)) for u, v in data.val_dict.items()}) if vr > 0 else test
    dump(os.path.join(d, "traindata.json"), rows(data.R))
    dump(os.path.join(d, "valdata.json"), val)
    dump(os.path.join(d, "testdata.json"), test)
    dump(os.path.join(d, "user_users_d.json"), social(data))
    meta[tag] = dict(num_user=int(data.user_num), num_item=int(data.item_num), train_pairs=int(data.R.nnz),
                     val_users=len(val) if vr > 0 else 0, test_users=len(test), summary=data.summary())
json.dump(meta, open(os.path.join(out, ds, "meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
