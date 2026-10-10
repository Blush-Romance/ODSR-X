"""Dataset loading.

ID mapping, test filtering and graph normalisation follow the upstream ``source/common.py``
exactly (users/items numbered in order of first appearance in train.txt; test pairs kept only
when both ids appear in train), so checkpoints and metrics are interchangeable with ODSR.
"""
import os
import re
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch


def _read_pairs(path: str) -> List[tuple]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [tuple(re.split(r"[\s,]+", line.strip())[:2]) for line in f if line.strip()]


def _to_torch_sparse(m: sp.spmatrix, device: torch.device) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack((m.row, m.col)).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data.astype(np.float32)), m.shape).coalesce().to(device)


class SocialRecData:
    """Fields used by the models: user_num, item_num, train_data, train_items, test_dict,
    adj_ui (U x I binary), norm_adj (LightGCN bipartite, sym-norm, no self loop),
    adj_uu (social, row-normalised, with self loop), R (scipy csr of train interactions)."""

    def __init__(self, root: str, dataset: str, device: torch.device,
                 val_ratio: float = 0.0, val_seed: int = 0, val_min_items: int = 5):
        base = os.path.join(root, dataset)
        raw_train = _read_pairs(os.path.join(base, "train.txt"))
        raw_test = _read_pairs(os.path.join(base, "test.txt"))
        raw_social = _read_pairs(os.path.join(base, "trust.txt"))

        u_map: Dict[str, int] = {}
        i_map: Dict[str, int] = {}
        train = np.array([(u_map.setdefault(u, len(u_map)), i_map.setdefault(i, len(i_map)))
                          for u, i in raw_train], dtype=np.int64)
        self.user_num, self.item_num = len(u_map), len(i_map)
        self.dataset, self.device = dataset, device

        test_dict: Dict[int, List[int]] = {}
        for u, i in raw_test:
            if u in u_map and i in i_map:
                test_dict.setdefault(u_map[u], []).append(i_map[i])

        # optional validation split carved out of train (per user, for hyper-parameter selection)
        self.val_dict: Dict[int, List[int]] = {}
        if val_ratio > 0:
            rng = np.random.default_rng(val_seed)
            train = train[rng.permutation(len(train))]
            train = train[np.argsort(train[:, 0], kind="stable")]
            _, start, cnt = np.unique(train[:, 0], return_index=True, return_counts=True)
            is_val = np.zeros(len(train), bool)
            for s0, c in zip(start, cnt):
                if c >= val_min_items:
                    is_val[s0:s0 + int(round(val_ratio * c))] = True
            for u, i in train[is_val]:
                self.val_dict.setdefault(int(u), []).append(int(i))
            train = train[~is_val]
        self.test_dict = test_dict
        # in validation mode the model is selected on val; test is still reported for reference
        self.eval_dict = self.val_dict if val_ratio > 0 else self.test_dict

        self.train_data = train
        self.train_items: Dict[int, List[int]] = {}
        for u, i in train:
            self.train_items.setdefault(int(u), []).append(int(i))

        U, I = self.user_num, self.item_num
        R = sp.csr_matrix((np.ones(len(train), np.float32), (train[:, 0], train[:, 1])), shape=(U, I))
        R.data[:] = 1.0
        self.R = R
        self.adj_ui = _to_torch_sparse(R, device)

        bip = sp.bmat([[None, R], [R.T, None]]).tocsr()
        deg = np.asarray(bip.sum(1)).ravel()
        d = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
        self.norm_adj = _to_torch_sparse(sp.diags(d) @ bip @ sp.diags(d), device)

        edges = set()
        for u, v in raw_social:
            if u in u_map and v in u_map and u_map[u] != u_map[v]:
                a, b = sorted((u_map[u], u_map[v]))
                edges.add((a, b))
        e = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
        rows = np.r_[e[:, 0], e[:, 1], np.arange(U)]
        cols = np.r_[e[:, 1], e[:, 0], np.arange(U)]
        S = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(U, U))
        S.data[:] = 1.0
        S = sp.diags(1.0 / np.asarray(S.sum(1)).ravel()) @ S
        self.adj_uu = _to_torch_sparse(S, device)
        self.social_edges = len(e)

    def summary(self) -> str:
        return (f"{self.dataset}: users={self.user_num} items={self.item_num} train={len(self.train_data)} "
                f"eval_users={len(self.eval_dict)} test_users={len(self.test_dict)} social={self.social_edges}")
