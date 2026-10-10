"""ODSR social graph backbone.

Copied from the official KDD'26 implementation (github.com/Rongfeng-Guo/ODSR-KDD-2026, ODSR.py),
restructured only (state-dict keys unchanged, so the official checkpoints load strictly).
Per layer: LightGCN item propagation; on the user side the aggregated social message is split
into the component parallel to the user's item message and the orthogonal remainder, and the
orthogonal part is admitted through a learned vector gate. SSL perturbs the orthogonal direction
with magnitude (1 - gate) * eps and contrasts it with the clean view (InfoNCE).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ODSRLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.W_gate = nn.Linear(3 * dim + 2, dim)
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.constant_(self.W_gate.bias, -0.5)

    def forward(self, h_i, s_u, adj_ui, adj_uu, ssl: bool, eps: float):
        m_item = self.dropout(torch.sparse.mm(adj_ui, h_i))
        m_social = self.dropout(torch.sparse.mm(adj_uu, s_u))
        item_sq = (m_item * m_item).sum(1, keepdim=True) + 1e-7
        dot = (m_social * m_item).sum(1, keepdim=True)
        s_par = dot / item_sq * m_item
        s_orth = m_social - s_par
        social_norm = m_social.norm(dim=1, keepdim=True) + 1e-7
        e_cos = dot / (social_norm * torch.sqrt(item_sq))
        e_norm = torch.tanh(social_norm)
        gate = torch.sigmoid(self.W_gate(torch.cat([m_item, s_orth, m_item * s_orth, e_cos, e_norm], 1)))
        s_clean = s_par + gate * s_orth
        h_u = m_item + s_clean
        if ssl:
            direction = s_orth / (s_orth.norm(dim=1, keepdim=True) + 1e-10)
            h_u = h_u + direction * (1.0 - gate.detach()) * eps
        return h_u, s_clean


class ODSRBackbone(nn.Module):
    def __init__(self, data, emb_size: int = 64, n_layer: int = 3, dropout: float = 0.0):
        super().__init__()
        self.user_num, self.item_num = data.user_num, data.item_num
        self.adj_ui, self.adj_uu, self.norm_adj = data.adj_ui, data.adj_uu, data.norm_adj
        self.layers = nn.ModuleList([ODSRLayer(emb_size, dropout) for _ in range(n_layer)])
        self.embedding_dict = nn.ParameterDict({
            "user_emb": nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.user_num, emb_size))),
            "item_emb": nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.item_num, emb_size))),
        })
        self.alphas = nn.Parameter(torch.ones(n_layer + 1))

    def forward(self, ssl: bool = False, eps: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        h_u, h_i = self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]
        s_u = h_u
        us, its = [h_u], [h_i]
        for layer in self.layers:
            side = torch.sparse.mm(self.norm_adj, torch.cat([h_u, h_i], 0))
            h_i_new = side[self.user_num:]
            h_u, s_u = layer(h_i, s_u, self.adj_ui, self.adj_uu, ssl, eps)
            h_i = h_i_new
            us.append(h_u)
            its.append(h_i)
        w = F.softmax(self.alphas, 0).view(1, -1, 1)
        return (torch.stack(us, 1) * w).sum(1), (torch.stack(its, 1) * w).sum(1)
