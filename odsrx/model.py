"""ODSR-X: social graph backbone that learns the residual of a frozen closed-form item-item prior.

score(u, i) = <e_u, e_i> + alpha * z_u(i)
  e_u, e_i : ODSR backbone embeddings (trainable)
  z_u(i)   : per-user standardised EASE-family score (fixed, computed once in closed form)
  alpha    : learnable scalar (log-parameterised, > 0)
With prior=None this is exactly ODSR (same loss, same SSL), which is the in-pipeline baseline.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ODSRBackbone


class HybridODSR(nn.Module):
    def __init__(self, data, prior: Optional[torch.Tensor], emb_size=64, n_layer=3, dropout=0.0,
                 reg=1e-5, ssl_reg=0.5, eps=0.8, tau=0.2, alpha_init=1.0, learn_alpha=True):
        super().__init__()
        self.backbone = ODSRBackbone(data, emb_size, n_layer, dropout)
        self.reg, self.ssl_reg, self.eps, self.tau = reg, ssl_reg, eps, tau
        self.prior = prior  # dense U x I tensor on CPU/GPU, not a parameter
        self.log_alpha = nn.Parameter(torch.tensor(float(alpha_init)).log(), requires_grad=learn_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _prior(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if self.prior is None:
            return 0.0
        return self.prior[users.to(self.prior.device), items.to(self.prior.device)].to(users.device)

    def compute_loss(self, users, pos, neg):
        fu, fi = self.backbone(ssl=False)
        u_e, p_e, n_e = fu[users], fi[pos], fi[neg]
        s_pos = (u_e * p_e).sum(1) + self.alpha * self._prior(users, pos)
        s_neg = (u_e * n_e).sum(1) + self.alpha * self._prior(users, neg)
        rec = -torch.log(1e-8 + torch.sigmoid(s_pos - s_neg)).mean()
        reg = self.reg * (u_e.norm(2) + p_e.norm(2) + n_e.norm(2))
        ssl = torch.zeros((), device=fu.device)
        if self.ssl_reg > 0:
            v1, _ = self.backbone(ssl=True, eps=self.eps)
            v2, _ = self.backbone(ssl=False)
            v1, v2 = F.normalize(v1[users], dim=1), F.normalize(v2[users], dim=1)
            logits = v1 @ v2.t() / self.tau
            ssl = self.ssl_reg * F.cross_entropy(logits, torch.arange(len(users), device=logits.device))
        return rec + reg + ssl, {"rec": rec.item(), "ssl": ssl.item(), "alpha": self.alpha.item()}

    @torch.no_grad()
    def context(self):
        return self.backbone(ssl=False)

    @torch.no_grad()
    def score_users(self, users: torch.Tensor, ctx) -> torch.Tensor:
        fu, fi = ctx
        s = fu[users] @ fi.t()
        if self.prior is not None:
            s = s + self.alpha * self.prior[users.to(self.prior.device)].to(s.device)
        return s
