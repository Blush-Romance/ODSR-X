"""Closed-form item-item prior (EASE family).

B = argmin ||X - X B||^2 + ||Lambda^{1/2} B||^2  s.t. diag(B) = 0
  -> P = (G + Lambda)^{-1},  B = -P / diag(P),  B_jj = 0          (Steck, WWW 2019)
Lambda = lam * I + p / (1 - p) * diag(G)   adds the dropout-denoising penalty of Steck (NeurIPS 2020).
beta > 0 runs the regression on the degree-normalised Gram D^{-beta/2} G D^{-beta/2}
and maps B back to raw inputs, which down-weights very popular items.
"""
import hashlib
import os

import numpy as np
import scipy.sparse as sp
import torch


def fit_item_prior(R: sp.csr_matrix, lam: float, p: float = 0.0, beta: float = 0.0) -> torch.Tensor:
    G = torch.from_numpy((R.T @ R).toarray().astype(np.float32))
    d = G.diagonal().clone()
    if beta > 0:
        s = d.clamp(min=1).pow(-beta / 2)
        G.mul_(s[:, None]).mul_(s[None, :])
        d = G.diagonal().clone()
    G.diagonal().add_(lam + (p / (1 - p)) * d)
    P = torch.linalg.inv(G)
    del G
    P.div_(-P.diagonal().clone()[None, :])
    P.fill_diagonal_(0.0)
    if beta > 0:
        s = torch.from_numpy(np.asarray(R.sum(0)).ravel().astype(np.float32)).clamp(min=1).pow(beta / 2)
        P.mul_(1.0 / s[:, None]).mul_(s[None, :])
    return P


def item_prior_scores(R: sp.csr_matrix, lam: float, p: float, beta: float, cache_dir: str = None,
                      standardize: bool = True) -> torch.Tensor:
    """Dense U x I prior scores X B, optionally z-scored per user over the full catalogue.
    Cached on disk keyed by the interaction matrix and the hyper-parameters."""
    key = hashlib.sha1(np.concatenate([R.indptr, R.indices]).astype(np.int64).tobytes()
                       + f"{R.shape}|{lam}|{p}|{beta}|{standardize}".encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"prior_{key}.npy") if cache_dir else None
    if path and os.path.exists(path):
        return torch.from_numpy(np.load(path))
    B = fit_item_prior(R, lam, p, beta)
    X = torch.from_numpy(R.toarray())
    S = torch.empty(R.shape, dtype=torch.float32)
    for st in range(0, R.shape[0], 4096):
        S[st:st + 4096] = X[st:st + 4096] @ B
    del B, X
    if standardize:
        S.sub_(S.mean(1, keepdim=True)).div_(S.std(1, keepdim=True) + 1e-8)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, S.numpy())
    return S


def social_scores(R: sp.csr_matrix, adj_uu: torch.Tensor, standardize: bool = True) -> torch.Tensor:
    """Dense U x I social co-consumption view: mean of friends' interaction rows
    (binary friendship, self loop removed, row-normalised), z-scored per user."""
    a = adj_uu.coalesce()
    idx = a.indices().cpu().numpy()
    S = sp.csr_matrix((np.ones(idx.shape[1], np.float32), (idx[0], idx[1])), shape=(R.shape[0],) * 2)
    S.setdiag(0)
    S.eliminate_zeros()
    deg = np.asarray(S.sum(1)).ravel()
    S = sp.diags(1.0 / np.maximum(deg, 1)) @ S
    SR = (S @ R.astype(np.float32)).tocsr()
    out = torch.empty(R.shape, dtype=torch.float32)
    for st in range(0, R.shape[0], 4096):
        out[st:st + 4096] = torch.from_numpy(SR[st:st + 4096].toarray())
    if standardize:
        out.sub_(out.mean(1, keepdim=True)).div_(out.std(1, keepdim=True) + 1e-8)
    return out
