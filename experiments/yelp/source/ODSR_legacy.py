# -*- coding: utf-8 -*-
"""
ODSR.py
Unified "keep useful parts" version (code-as-truth):

- social state update: s^{(l+1)} = s_clean^{(l)}
- vector gate (dim-wise)
- micro-interact: m_u_item ⊙ s_orth
- geo evidence: e_cos, e_norm (broadcast scalars)
- risk-aware SSL:
    adaptive noise uses (1 - stopgrad(gate)) as magnitude mask (element-wise)
    direction: normalized s_orth
    gate is stop-grad in SSL path via detach()
"""

from typing import Any, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_negative_bpr_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    """BPR over all supplied negatives or only the maximum-scoring one."""
    if neg_score.dim() != 2:
        raise ValueError("neg_score must have shape [batch, negative_k]")
    if strategy in {"random_mean", "hard_topk_mean"}:
        return -F.logsigmoid(pos_score.unsqueeze(1) - neg_score).mean()
    if strategy == "hard_max":
        hard_neg_score = neg_score.max(dim=1).values
        return -F.logsigmoid(pos_score - hard_neg_score).mean()
    raise ValueError(f"unsupported negative_strategy: {strategy}")


class ODSRLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        candidate_residual_bound: float,
        item_residual_bound: float,
        conditional_residual_bound: float,
        normalized_cross_scale: float,
        item_cosine_groups: int,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.candidate_residual_bound = float(candidate_residual_bound)
        if self.candidate_residual_bound <= 0.0:
            raise ValueError("candidate_residual_bound must be > 0")
        self.item_residual_bound = float(item_residual_bound)
        self.conditional_residual_bound = float(conditional_residual_bound)
        if self.item_residual_bound <= 0.0:
            raise ValueError("item_residual_bound must be > 0")
        if self.conditional_residual_bound <= 0.0:
            raise ValueError("conditional_residual_bound must be > 0")
        self.normalized_cross_scale = float(normalized_cross_scale)
        if self.normalized_cross_scale < 0.0:
            raise ValueError("normalized_cross_scale must be >= 0")
        self.item_cosine_groups = int(item_cosine_groups)
        if self.item_cosine_groups < 1:
            raise ValueError("item_cosine_groups must be >= 1")
        if dim % self.item_cosine_groups != 0:
            raise ValueError("embedding dimension must divide item_cosine_groups")

        # gate input: [m_u_item, s_orth, (m_u_item ⊙ s_orth), e_cos, e_norm]
        dim_input = dim + dim + dim + 2
        self.W_gate = nn.Linear(dim_input, dim)

        # V1 candidate-aware residual gate. The global gate is still used for
        # propagation; a candidate only re-filters s_orth during readout.
        self.W_item_gate = nn.Linear(dim, dim, bias=False)
        self.item_cross_scale = nn.Parameter(torch.zeros(dim))
        self.item_cos_scale = nn.Parameter(torch.zeros(dim))

        # Independent evidence branch for probability-space interpolation.
        # It estimates how much the candidate proposal should be trusted;
        # unlike the proposal itself, zero evidence maps to a low confidence
        # through a fixed negative logit bias supplied by the model config.
        confidence_rng_state = torch.get_rng_state()
        self.W_item_confidence = nn.Linear(dim, dim, bias=False)
        torch.set_rng_state(confidence_rng_state)
        self.item_confidence_cross_scale = nn.Parameter(torch.zeros(dim))
        self.item_confidence_cos_scale = nn.Parameter(torch.zeros(dim))

        # Independent candidate-conditioned scorer for information that the
        # original ODSR gate retained but the current candidate may reject.
        # Restoring the RNG keeps all pre-existing initializations identical
        # in same-seed ablations.
        suppress_rng_state = torch.get_rng_state()
        self.W_item_suppress = nn.Linear(dim, dim, bias=False)
        torch.set_rng_state(suppress_rng_state)
        self.item_suppress_cross_scale = nn.Parameter(torch.zeros(dim))
        self.item_suppress_cos_scale = nn.Parameter(torch.zeros(dim))

        # Learn an aligned space for candidate/social dimension matching.
        # Identity initialization makes the projected-cross ablation exactly
        # match the bounded normalized-cross variant at initialization.  RNG
        # restoration preserves every pre-existing parameter initialization.
        match_rng_state = torch.get_rng_state()
        self.W_item_match = nn.Linear(dim, dim, bias=False)
        self.W_social_match = nn.Linear(dim, dim, bias=False)
        torch.set_rng_state(match_rng_state)
        with torch.no_grad():
            nn.init.eye_(self.W_item_match.weight)
            nn.init.eye_(self.W_social_match.weight)

        # Independent branch for a user-candidate conditional-interest
        # residual. Its RNG use is restored so older ablations retain exactly
        # the same backbone initialization.
        conditional_rng_state = torch.get_rng_state()
        self.W_conditional_gate = nn.Linear(dim, dim, bias=False)
        torch.set_rng_state(conditional_rng_state)
        self.conditional_cross_scale = nn.Parameter(torch.zeros(dim))
        self.conditional_cos_scale = nn.Parameter(torch.zeros(dim))
        # Per-dimension fusion: sigmoid(0) = 0.5 at initialization.
        self.dual_residual_mix_logit = nn.Parameter(torch.zeros(dim))

        # Candidate-conditioned decomposition anchor.  This moves item-aware
        # conditioning one step before the orthogonal gate: the same aggregated
        # social message is re-split into parallel/orthogonal components for
        # every candidate.  Zero initialization keeps the initial anchor equal
        # to the original ODSR item-interest message.
        # Constructing nn.Linear consumes RNG even when its weights are zeroed.
        # Restore the CPU RNG afterwards so adding this branch does not change
        # the original ODSR/gate/embedding initialization for fair ablations.
        anchor_rng_state = torch.get_rng_state()
        self.W_item_anchor = nn.Linear(dim, dim, bias=False)
        torch.set_rng_state(anchor_rng_state)
        self.item_anchor_cross_scale = nn.Parameter(torch.zeros(dim))
        self.item_anchor_cos_scale = nn.Parameter(torch.zeros(dim))

        # Pure user-item interaction branch.  The candidate has no standalone
        # path into the residual: it first interacts element-wise with the
        # user's item-interest message, and the resulting adjustment is
        # applied back to that same user-interest message.
        interaction_rng_state = torch.get_rng_state()
        self.W_interaction_adjust = nn.Linear(dim, dim, bias=False)
        torch.set_rng_state(interaction_rng_state)

        # Exact interaction-to-gate residual requested for the clean ablation:
        # m_u * e_i -> W_x -> eta * tanh, with no second multiplication by
        # m_u and no additional residual-gate mapping.
        direct_interaction_rng_state = torch.get_rng_state()
        self.W_interaction_gate = nn.Linear(dim, dim, bias=False)
        torch.set_rng_state(direct_interaction_rng_state)

        # Candidate-conditioned orthogonal-social representation used by the
        # symmetric user/social interaction ablation.
        social_interaction_rng_state = torch.get_rng_state()
        self.W_social_interaction = nn.Linear(dim, dim, bias=False)
        torch.set_rng_state(social_interaction_rng_state)

        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.constant_(self.W_gate.bias, -0.5)
        # V1 starts exactly from the global ODSR gate.
        nn.init.zeros_(self.W_item_gate.weight)
        nn.init.zeros_(self.W_item_confidence.weight)
        nn.init.zeros_(self.W_item_suppress.weight)
        nn.init.zeros_(self.W_conditional_gate.weight)
        nn.init.zeros_(self.W_item_anchor.weight)
        nn.init.eye_(self.W_interaction_adjust.weight)
        # Zero residual at initialization preserves the original ODSR gate;
        # this layer receives gradients directly from the final gate.
        nn.init.zeros_(self.W_interaction_gate.weight)
        nn.init.eye_(self.W_social_interaction.weight)

    def forward(
        self,
        h_i: torch.Tensor,
        s_u: torch.Tensor,
        adj_ui: torch.Tensor,
        adj_uu: torch.Tensor,
        ssl_mode: Optional[str],
        eps: float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns:
          h_u_new, s_clean, gate, gate_logit, base, s_orth,
          m_u_item, m_u_social
        """
        # item-view anchor message
        m_u_item = self.dropout(torch.sparse.mm(adj_ui, h_i))
        # social message from social state
        m_u_social = self.dropout(torch.sparse.mm(adj_uu, s_u))

        # projection decomposition (aggregate-then-decompose)
        item_norm_sq = (m_u_item * m_u_item).sum(dim=1, keepdim=True) + 1e-7
        dot = (m_u_social * m_u_item).sum(dim=1, keepdim=True)
        proj_coeff = dot / item_norm_sq
        s_parallel = proj_coeff * m_u_item
        s_orth = m_u_social - s_parallel

        # geo evidence (broadcast scalars)
        social_norm = m_u_social.norm(dim=1, keepdim=True) + 1e-7
        item_norm = torch.sqrt(item_norm_sq)
        e_cos = dot / (social_norm * item_norm)     # [U,1]
        e_norm = torch.tanh(social_norm)            # [U,1]

        # micro-interact
        micro = m_u_item * s_orth                   # [U,D]

        gate_in = torch.cat([m_u_item, s_orth, micro, e_cos, e_norm], dim=1)
        gate_logit = self.W_gate(gate_in)
        gate = torch.sigmoid(gate_logit)            # vector gate [U,D]

        s_clean = s_parallel + gate * s_orth
        base = m_u_item + s_parallel
        h_u_new = base + gate * s_orth

        # risk-aware adaptive view: element-wise magnitude mask (1 - sg(g))
        if ssl_mode == "adaptive":
            g_det = gate.detach()
            mag = (1.0 - g_det) * eps              # [U,D]
            orth_norm = s_orth.norm(dim=1, keepdim=True) + 1e-10
            direction = s_orth / orth_norm
            noise = direction * mag
            h_u_new = h_u_new + noise

        return (
            h_u_new,
            s_clean,
            gate,
            gate_logit,
            base,
            s_orth,
            m_u_item,
            m_u_social,
        )

    def item_aware_gate(
        self,
        global_gate: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Candidate-conditioned dimension gate with broadcastable inputs.

        Pair scoring uses [B, D] inputs.  Full ranking uses [B, 1, D]
        user tensors and a [1, C, D] candidate tensor for an item chunk.
        """
        item_delta = self.item_aware_gate_delta(s_orth, candidate)
        global_gate = global_gate.expand_as(item_delta)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        return torch.sigmoid(global_logit + item_delta)

    def item_aware_gate_delta(
        self,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unbounded candidate residual in gate-logit space."""
        # Project candidates before Cartesian broadcasting so full ranking
        # pays O(C * D^2), rather than O(B * C * D^2), for this term.
        item_linear = self.W_item_gate(candidate)
        s_orth, candidate = torch.broadcast_tensors(s_orth, candidate)
        item_linear = item_linear.expand_as(s_orth)
        candidate_cos = F.cosine_similarity(
            candidate, s_orth, dim=-1
        ).unsqueeze(-1)
        return (
            item_linear
            + self.item_cross_scale * (candidate * s_orth)
            + self.item_cos_scale * candidate_cos
        )

    def item_aware_confidence_direct_gate(
        self,
        global_gate: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
        confidence_logit_bias: float,
        return_parts: bool = False,
    ):
        """Blend the ODSR gate with a candidate proposal in probability space.

        Both the proposal and confidence use the original random-K8 matching
        inputs: candidate, full orthogonal social information, their
        element-wise interaction, and cosine similarity. The final gate is
        ``g_ui = (1-confidence) * g + confidence * proposal``.

        A fixed negative confidence bias makes no-evidence confidence close
        to zero, while retaining non-zero gradients for both branches.
        """
        proposal_raw = self.item_aware_gate_delta(s_orth, candidate)
        proposal = torch.sigmoid(proposal_raw)

        confidence_linear = self.W_item_confidence(candidate)
        s_orth, candidate = torch.broadcast_tensors(s_orth, candidate)
        confidence_linear = confidence_linear.expand_as(s_orth)
        candidate_cos = F.cosine_similarity(
            candidate, s_orth, dim=-1
        ).unsqueeze(-1)
        confidence_raw = (
            confidence_linear
            + self.item_confidence_cross_scale * (candidate * s_orth)
            + self.item_confidence_cos_scale * candidate_cos
        )
        confidence = torch.sigmoid(
            confidence_raw + float(confidence_logit_bias)
        )
        global_gate = global_gate.expand_as(proposal)
        gate_ui = (
            (1.0 - confidence) * global_gate
            + confidence * proposal
        )
        if return_parts:
            return gate_ui, proposal, confidence, proposal_raw, confidence_raw
        return gate_ui

    def item_aware_signed_direct_gate(
        self,
        global_gate: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
        signed_scale: float,
        return_parts: bool = False,
    ):
        """Apply one zero-centered signed correction in gate space.

        Positive ratios consume only the remaining ``1-g`` capacity, while
        negative ratios remove only a fraction of the current ``g``. For a
        scale in [0, 1], the corrected gate therefore remains in [0, 1].
        """
        raw_match = self.item_aware_gate_delta(s_orth, candidate)
        signed_ratio = float(signed_scale) * torch.tanh(raw_match)
        global_gate = global_gate.expand_as(signed_ratio)
        gate_change = torch.where(
            signed_ratio >= 0.0,
            (1.0 - global_gate) * signed_ratio,
            global_gate * signed_ratio,
        )
        gate_ui = global_gate + gate_change
        if return_parts:
            return gate_ui, gate_change, signed_ratio, raw_match
        return gate_ui

    def item_aware_dropped_recovery_gate(
        self,
        global_gate: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
        recovery_scale: float,
    ) -> torch.Tensor:
        """Recover candidate-useful information rejected by the ODSR gate.

        The candidate branch only observes
        ``s_drop = (1 - global_gate) * s_orth``.  Its recovery probability can
        only add back a bounded fraction of this rejected information:

        ``gate_ui = global_gate + lambda * (1-global_gate) * sigmoid(delta)``.
        """
        s_drop = (1.0 - global_gate) * s_orth
        raw_delta = self.item_aware_gate_delta(s_drop, candidate)
        recovery = torch.sigmoid(raw_delta)
        global_gate = global_gate.expand_as(recovery)
        return global_gate + (
            float(recovery_scale) * (1.0 - global_gate) * recovery
        )

    def item_aware_bidirectional_gate(
        self,
        global_gate: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
        correction_scale: float,
        return_parts: bool = False,
    ):
        """Directly recover rejected information and suppress retained noise.

        The two scorers receive disjoint parts of the original orthogonal
        social message:

        ``s_drop = (1-g) * s_orth`` and ``s_keep = g * s_orth``.

        For ``lambda`` in [0, 1], the resulting gate remains in [0, 1]:

        ``g_ui = g + lambda*(1-g)*r_plus - lambda*g*r_minus``.
        """
        s_drop = (1.0 - global_gate) * s_orth
        s_keep = global_gate * s_orth

        recover_raw = self.item_aware_gate_delta(s_drop, candidate)

        suppress_linear = self.W_item_suppress(candidate)
        s_keep, candidate = torch.broadcast_tensors(s_keep, candidate)
        suppress_linear = suppress_linear.expand_as(s_keep)
        keep_cos = F.cosine_similarity(
            candidate, s_keep, dim=-1
        ).unsqueeze(-1)
        suppress_raw = (
            suppress_linear
            + self.item_suppress_cross_scale * (candidate * s_keep)
            + self.item_suppress_cos_scale * keep_cos
        )

        recover = torch.sigmoid(recover_raw)
        suppress = torch.sigmoid(suppress_raw)
        global_gate = global_gate.expand_as(recover)
        gate_ui = (
            global_gate
            + float(correction_scale) * (1.0 - global_gate) * recover
            - float(correction_scale) * global_gate * suppress
        )
        if return_parts:
            return gate_ui, recover, suppress, recover_raw, suppress_raw
        return gate_ui

    def item_aware_gate_normalized_cross(
        self,
        global_gate: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
        bound_cross: bool = False,
    ) -> torch.Tensor:
        """Candidate residual with RMS-normalized element-wise matching.

        Only the cross term uses normalized inputs.  When ``bound_cross`` is
        true, tanh removes its long tail and ``normalized_cross_scale`` limits
        its amplitude.  The candidate linear term, cosine term, global user
        gate, and final logit-space addition remain unchanged.
        """
        item_linear = self.W_item_gate(candidate)
        s_orth, candidate = torch.broadcast_tensors(s_orth, candidate)
        item_linear = item_linear.expand_as(s_orth)

        candidate_rms = torch.sqrt(
            candidate.pow(2).mean(dim=-1, keepdim=True) + 1e-8
        )
        orth_rms = torch.sqrt(
            s_orth.pow(2).mean(dim=-1, keepdim=True) + 1e-8
        )
        normalized_cross = (
            candidate / candidate_rms
        ) * (
            s_orth / orth_rms
        )
        if bound_cross:
            normalized_cross = (
                self.normalized_cross_scale * torch.tanh(normalized_cross)
            )
        candidate_cos = F.cosine_similarity(
            candidate, s_orth, dim=-1
        ).unsqueeze(-1)
        item_delta = (
            item_linear
            + self.item_cross_scale * normalized_cross
            + self.item_cos_scale * candidate_cos
        )
        global_gate = global_gate.expand_as(item_delta)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        return torch.sigmoid(global_logit + item_delta)

    def item_aware_gate_projected_cross(
        self,
        global_gate: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Bounded matching after learning a shared item/social space.

        The two identity-initialized projections can learn how original item
        and social coordinates should be aligned.  RMS normalization, tanh,
        and ``normalized_cross_scale`` retain the stable amplitude of the
        preceding bounded normalized-cross ablation.
        """
        item_linear = self.W_item_gate(candidate)
        projected_candidate = self.W_item_match(candidate)
        projected_social = self.W_social_match(s_orth)

        projected_candidate = projected_candidate / torch.sqrt(
            projected_candidate.pow(2).mean(dim=-1, keepdim=True) + 1e-8
        )
        projected_social = projected_social / torch.sqrt(
            projected_social.pow(2).mean(dim=-1, keepdim=True) + 1e-8
        )
        projected_social, projected_candidate = torch.broadcast_tensors(
            projected_social, projected_candidate
        )
        matched_cross = self.normalized_cross_scale * torch.tanh(
            projected_candidate * projected_social
        )

        s_orth, candidate = torch.broadcast_tensors(s_orth, candidate)
        item_linear = item_linear.expand_as(s_orth)
        candidate_cos = F.cosine_similarity(
            candidate, s_orth, dim=-1
        ).unsqueeze(-1)
        item_delta = (
            item_linear
            + self.item_cross_scale * matched_cross
            + self.item_cos_scale * candidate_cos
        )
        global_gate = global_gate.expand_as(item_delta)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        return torch.sigmoid(global_logit + item_delta)

    def item_aware_gate_group_cosine(
        self,
        global_gate: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Candidate linear residual plus group-wise cosine matching.

        The unstable raw element-wise cross branch is deliberately absent.
        Grouped cosine retains scale stability while exposing which regions
        of the shared embedding space match the candidate.
        """
        item_linear = self.W_item_gate(candidate)
        s_orth, candidate = torch.broadcast_tensors(s_orth, candidate)
        item_linear = item_linear.expand_as(s_orth)

        group_dim = candidate.size(-1) // self.item_cosine_groups
        candidate_grouped = candidate.reshape(
            *candidate.shape[:-1], self.item_cosine_groups, group_dim
        )
        orth_grouped = s_orth.reshape(
            *s_orth.shape[:-1], self.item_cosine_groups, group_dim
        )
        group_cosine = F.cosine_similarity(
            candidate_grouped, orth_grouped, dim=-1
        )
        group_cosine = group_cosine.repeat_interleave(group_dim, dim=-1)

        item_delta = item_linear + self.item_cos_scale * group_cosine
        global_gate = global_gate.expand_as(item_delta)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        return torch.sigmoid(global_logit + item_delta)

    def conditional_gate_delta(
        self,
        s_orth: torch.Tensor,
        candidate_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Independent gate-logit residual for q_(u|i) - m_u."""
        conditional_linear = self.W_conditional_gate(candidate_delta)
        s_orth, candidate_delta = torch.broadcast_tensors(
            s_orth, candidate_delta
        )
        conditional_linear = conditional_linear.expand_as(s_orth)
        conditional_cos = F.cosine_similarity(
            candidate_delta, s_orth, dim=-1
        ).unsqueeze(-1)
        return (
            conditional_linear
            + self.conditional_cross_scale * (candidate_delta * s_orth)
            + self.conditional_cos_scale * conditional_cos
        )

    def candidate_redecompose_social(
        self,
        item_interest: torch.Tensor,
        social_message: torch.Tensor,
        global_gate: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-split one aggregated social message for each candidate.

        Inputs may be aligned [B, D] pairs or broadcastable full-ranking views
        such as [B, 1, D] users and [1, C, D] candidates.
        """
        candidate_view = candidate
        global_gate_view = global_gate
        candidate_anchor = self.candidate_interest_anchor(
            item_interest, candidate_view
        )
        candidate_anchor, social_message = torch.broadcast_tensors(
            candidate_anchor, social_message
        )

        anchor_norm_sq = (
            candidate_anchor * candidate_anchor
        ).sum(dim=-1, keepdim=True) + 1e-7
        social_anchor_dot = (
            social_message * candidate_anchor
        ).sum(dim=-1, keepdim=True)
        s_parallel = (
            social_anchor_dot / anchor_norm_sq
        ) * candidate_anchor
        s_orth = social_message - s_parallel

        # The original global gate remains the stable prior.  The existing
        # candidate residual gate then filters the newly defined orthogonal
        # component instead of the old candidate-independent one.
        gate_ui = self.item_aware_gate(
            global_gate_view, s_orth, candidate_view
        )
        useful_social = s_parallel + gate_ui * s_orth
        return useful_social, s_parallel, s_orth

    def candidate_interest_anchor(
        self,
        item_interest: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse user interest and a candidate without changing propagation."""
        return item_interest + self.candidate_interest_delta(
            item_interest, candidate
        )

    def candidate_interest_delta(
        self,
        item_interest: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the candidate-induced change to user interest.

        Unlike ``candidate_interest_anchor``, this tensor does not contain the
        base user interest itself. It is suitable as a residual-gate input when
        the global ODSR gate has already used ``item_interest``.
        """
        # Project candidates before Cartesian broadcasting so full ranking pays
        # O(C * D^2), rather than O(B * C * D^2), for this term.
        anchor_linear = self.W_item_anchor(candidate)
        item_interest, candidate = torch.broadcast_tensors(
            item_interest, candidate
        )
        anchor_linear = anchor_linear.expand_as(item_interest)
        candidate_interest_cos = F.cosine_similarity(
            candidate, item_interest, dim=-1
        ).unsqueeze(-1)
        anchor_delta = (
            anchor_linear
            + self.item_anchor_cross_scale * (candidate * item_interest)
            + self.item_anchor_cos_scale * candidate_interest_cos
        )
        return anchor_delta

    def candidate_interest_gate(
        self,
        item_interest: torch.Tensor,
        social_message: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
        detach_odsr_gate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use user-candidate conditional interest to filter fixed s_orth.

        The original user-interest decomposition remains unchanged.  Only the
        gate is recomputed under the candidate-conditioned interest anchor.
        """
        candidate_anchor = self.candidate_interest_anchor(
            item_interest, candidate
        )
        candidate_anchor, social_message, s_orth = torch.broadcast_tensors(
            candidate_anchor, social_message, s_orth
        )

        anchor_norm_sq = (
            candidate_anchor * candidate_anchor
        ).sum(dim=-1, keepdim=True) + 1e-7
        social_anchor_dot = (
            social_message * candidate_anchor
        ).sum(dim=-1, keepdim=True)
        social_norm = social_message.norm(dim=-1, keepdim=True) + 1e-7
        anchor_norm = torch.sqrt(anchor_norm_sq)
        e_cos = social_anchor_dot / (social_norm * anchor_norm)
        e_norm = torch.tanh(social_norm)
        micro = candidate_anchor * s_orth
        gate_in = torch.cat([
            candidate_anchor,
            s_orth,
            micro,
            e_cos,
            e_norm,
        ], dim=-1)
        if detach_odsr_gate:
            gate_logit = F.linear(
                gate_in,
                self.W_gate.weight.detach(),
                (
                    self.W_gate.bias.detach()
                    if self.W_gate.bias is not None
                    else None
                ),
            )
        else:
            gate_logit = self.W_gate(gate_in)
        gate_ui = torch.sigmoid(gate_logit)
        return gate_ui, candidate_anchor

    def candidate_interest_residual_gate(
        self,
        global_gate: torch.Tensor,
        item_interest: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adjust the global gate with user-candidate conditional interest.

        This preserves the original ODSR user gate as the prior and reuses the
        separate V1 residual branch.  The only change from V1 is replacing the
        raw candidate vector with the user-candidate conditional-interest
        anchor when matching against the fixed orthogonal social component.
        """
        candidate_anchor = self.candidate_interest_anchor(
            item_interest, candidate
        )
        gate_ui = self.item_aware_gate(
            global_gate, s_orth, candidate_anchor
        )
        return gate_ui, candidate_anchor

    def candidate_delta_residual_gate(
        self,
        global_gate: torch.Tensor,
        item_interest: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gate with only the candidate-induced conditional-interest change.

        The original gate already contains the base interest ``m_u``. Feeding
        the full ``q_{u|i} = m_u + r_{u|i}`` into the residual branch would use
        ``m_u`` a second time. This variant feeds only
        ``r_{u|i} = q_{u|i} - m_u`` into that branch.
        """
        candidate_delta = self.candidate_interest_delta(
            item_interest, candidate
        )
        raw_delta = self.item_aware_gate_delta(s_orth, candidate_delta)
        bounded_delta = self.candidate_residual_bound * torch.tanh(raw_delta)
        global_gate = global_gate.expand_as(bounded_delta)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        gate_ui = torch.sigmoid(global_logit + bounded_delta)
        return gate_ui, candidate_delta

    def candidate_interaction_residual_gate(
        self,
        global_gate: torch.Tensor,
        item_interest: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Residual gate from a pure user-candidate interaction adjustment.

        h_(u|i) = W_x(m_u * e_i)
        a_(u|i) = tanh(h_(u|i))
        r_(u|i) = m_u * a_(u|i)

        Only r_(u|i), never the raw candidate, enters the residual gate.  The
        final gate-logit correction keeps the same configured bound as the
        earlier single-branch conditional-residual ablation.
        """
        item_interest, candidate = torch.broadcast_tensors(
            item_interest, candidate
        )
        interaction = item_interest * candidate
        adjustment = torch.tanh(self.W_interaction_adjust(interaction))
        interest_delta = item_interest * adjustment

        raw_delta = self.item_aware_gate_delta(s_orth, interest_delta)
        bounded_delta = self.candidate_residual_bound * torch.tanh(raw_delta)
        global_gate = global_gate.expand_as(bounded_delta)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        gate_ui = torch.sigmoid(global_logit + bounded_delta)
        return gate_ui, interest_delta, adjustment

    def candidate_interaction_gate_residual(
        self,
        global_gate: torch.Tensor,
        item_interest: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map only m_u * e_i directly into a bounded gate-logit residual.

        q_(u|i) = W_x(m_u * e_i)
        a_(u|i) = eta * tanh(q_(u|i))
        g_(u|i) = sigmoid(logit(g_u) + a_(u|i))

        ``candidate_residual_bound`` is eta.  There is deliberately no
        r_(u|i) = m_u * a_(u|i), no raw-candidate path, and no second residual
        mapping or tanh.
        """
        item_interest, candidate = torch.broadcast_tensors(
            item_interest, candidate
        )
        interaction = item_interest * candidate
        gate_residual = self.candidate_residual_bound * torch.tanh(
            self.W_interaction_gate(interaction)
        )
        global_gate = global_gate.expand_as(gate_residual)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        gate_ui = torch.sigmoid(global_logit + gate_residual)
        return gate_ui, gate_residual

    def candidate_interaction_social_residual_gate(
        self,
        global_gate: torch.Tensor,
        item_interest: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter orthogonal social information with q_(u|i).

        q_(u|i) = W_x(m_u * e_i)
        raw_delta = W_q q_(u|i)
                    + c_q * (q_(u|i) * s_perp)
                    + d_q * cos(q_(u|i), s_perp)
        delta = eta * tanh(raw_delta)

        The candidate has no standalone path.  Its interaction with the user
        first forms q_(u|i), and q_(u|i) then matches the actual orthogonal
        social information using the original candidate-filter structure.
        """
        item_interest, candidate = torch.broadcast_tensors(
            item_interest, candidate
        )
        conditional_user = self.W_interaction_adjust(
            item_interest * candidate
        )
        raw_delta = self.item_aware_gate_delta(
            s_orth, conditional_user
        )
        gate_residual = self.candidate_residual_bound * torch.tanh(raw_delta)
        global_gate = global_gate.expand_as(gate_residual)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        gate_ui = torch.sigmoid(global_logit + gate_residual)
        return gate_ui, conditional_user, gate_residual

    def candidate_user_social_interaction_residual_gate(
        self,
        global_gate: torch.Tensor,
        item_interest: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Match candidate-conditioned user and orthogonal-social views.

        q_(u|i) = W_x(m_u * e_i)
        o_(u|i) = W_s(s_perp * e_i)
        raw_delta = W_q q_(u|i)
                    + c * (q_(u|i) * o_(u|i))
                    + d * cos(q_(u|i), o_(u|i))
        delta = eta * tanh(raw_delta)

        Both inputs to the residual matcher are candidate-conditioned.  The
        raw candidate has no standalone path into the gate residual.
        """
        item_interest, s_orth, candidate = torch.broadcast_tensors(
            item_interest, s_orth, candidate
        )
        conditional_user = self.W_interaction_adjust(
            item_interest * candidate
        )
        conditional_orth = self.W_social_interaction(
            s_orth * candidate
        )
        raw_delta = self.item_aware_gate_delta(
            conditional_orth, conditional_user
        )
        gate_residual = self.candidate_residual_bound * torch.tanh(raw_delta)
        global_gate = global_gate.expand_as(gate_residual)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        gate_ui = torch.sigmoid(global_logit + gate_residual)
        return gate_ui, conditional_user, conditional_orth, gate_residual

    def candidate_dual_residual_gate(
        self,
        global_gate: torch.Tensor,
        item_interest: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse raw-item and conditional-interest residual branches.

        One independent branch sees the raw candidate item. The other sees
        only q_(u|i) - m_u, so base user interest is not counted twice. Their
        gate-logit residuals are fused dimension-wise before one shared bound.
        """
        candidate_delta = self.candidate_interest_delta(
            item_interest, candidate
        )
        item_raw_delta = self.item_aware_gate_delta(s_orth, candidate)
        conditional_raw_delta = self.conditional_gate_delta(
            s_orth, candidate_delta
        )
        mix = torch.sigmoid(self.dual_residual_mix_logit)
        fused_raw_delta = (
            mix * item_raw_delta
            + (1.0 - mix) * conditional_raw_delta
        )
        bounded_delta = (
            self.candidate_residual_bound * torch.tanh(fused_raw_delta)
        )
        global_gate = global_gate.expand_as(bounded_delta)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        gate_ui = torch.sigmoid(global_logit + bounded_delta)
        return gate_ui, candidate_delta, mix

    def candidate_dual_separate_residual_gate(
        self,
        global_gate: torch.Tensor,
        item_interest: torch.Tensor,
        s_orth: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bound the raw-item and conditional residuals independently.

        This prevents either unbounded raw branch from dominating before the
        nonlinearity. The two already-bounded gate-logit residuals are summed,
        so the final absolute bound is item_bound + conditional_bound.
        """
        candidate_delta = self.candidate_interest_delta(
            item_interest, candidate
        )
        item_raw_delta = self.item_aware_gate_delta(s_orth, candidate)
        conditional_raw_delta = self.conditional_gate_delta(
            s_orth, candidate_delta
        )
        item_delta = self.item_residual_bound * torch.tanh(item_raw_delta)
        conditional_delta = (
            self.conditional_residual_bound
            * torch.tanh(conditional_raw_delta)
        )
        fused_delta = item_delta + conditional_delta
        global_gate = global_gate.expand_as(fused_delta)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1.0 - 1e-6))
        gate_ui = torch.sigmoid(global_logit + fused_delta)
        return gate_ui, candidate_delta, item_delta, conditional_delta


class ODSR(nn.Module):
    def __init__(self, data_obj, config: Dict, device: torch.device, ablation_flags: Optional[Dict] = None):
        super().__init__()
        self.device = device
        self.user_num = data_obj.user_num
        self.item_num = data_obj.item_num

        self.emb_size = int(config.get("emb_size", 64))
        self.n_layers = int(config.get("n_layer", 3))
        self.dropout_val = float(config.get("dropout", 0.0))

        self.reg = float(config["reg"])
        self.eps = float(config["eps"])
        self.ssl_reg = float(config["ssl_reg"])
        self.ssl_temp = float(config.get("tau", 0.2))
        self.negative_strategy = str(
            config.get("negative_strategy", "random_mean")
        )
        self.negative_k = int(config.get("negative_k", 8))
        self.negative_gradient_mode = str(
            config.get("negative_gradient_mode", "joint")
        )
        self.candidate_loss_weight = float(
            config.get("candidate_loss_weight", 1.0)
        )
        if self.negative_gradient_mode not in {"joint", "candidate_only"}:
            raise ValueError(
                "negative_gradient_mode must be 'joint' or 'candidate_only'"
            )
        if self.candidate_loss_weight < 0.0:
            raise ValueError("candidate_loss_weight must be >= 0")
        self.candidate_residual_bound = float(
            config.get("candidate_residual_bound", 0.25)
        )
        self.candidate_recovery_scale = float(
            config.get("candidate_recovery_scale", 0.1)
        )
        if not 0.0 <= self.candidate_recovery_scale <= 1.0:
            raise ValueError("candidate_recovery_scale must be in [0, 1]")
        self.candidate_confidence_bias = float(
            config.get("candidate_confidence_bias", -4.0)
        )
        if self.candidate_confidence_bias >= 0.0:
            raise ValueError("candidate_confidence_bias must be < 0")
        self.signed_gate_scale = float(config.get("signed_gate_scale", 0.2))
        self.gate_change_reg_weight = float(
            config.get("gate_change_reg_weight", 0.0)
        )
        if not 0.0 <= self.signed_gate_scale <= 1.0:
            raise ValueError("signed_gate_scale must be in [0, 1]")
        if self.gate_change_reg_weight < 0.0:
            raise ValueError("gate_change_reg_weight must be >= 0")
        self.gate_saturation_bound = float(
            config.get("gate_saturation_bound", 3.0)
        )
        self.gate_saturation_reg_weight = float(
            config.get("gate_saturation_reg_weight", 0.0)
        )
        if self.gate_saturation_bound <= 0.0:
            raise ValueError("gate_saturation_bound must be > 0")
        if self.gate_saturation_reg_weight < 0.0:
            raise ValueError("gate_saturation_reg_weight must be >= 0")
        self.confidence_warmup_epochs = int(
            config.get("confidence_warmup_epochs", 0)
        )
        self.proposal_warmup_loss_weight = float(
            config.get("proposal_warmup_loss_weight", 0.0)
        )
        self.proposal_aux_loss_weight = float(
            config.get("proposal_aux_loss_weight", 0.0)
        )
        if self.confidence_warmup_epochs < 0:
            raise ValueError("confidence_warmup_epochs must be >= 0")
        if self.proposal_warmup_loss_weight < 0.0:
            raise ValueError("proposal_warmup_loss_weight must be >= 0")
        if self.proposal_aux_loss_weight < 0.0:
            raise ValueError("proposal_aux_loss_weight must be >= 0")
        self.item_residual_bound = float(
            config.get("item_residual_bound", 0.1)
        )
        self.conditional_residual_bound = float(
            config.get("conditional_residual_bound", 0.1)
        )
        self.normalized_cross_scale = float(
            config.get("normalized_cross_scale", 0.05)
        )
        self.item_cosine_groups = int(config.get("item_cosine_groups", 4))
        self.item_branch_loss_weight = float(
            config.get("item_branch_loss_weight", 0.0)
        )
        self.conditional_branch_loss_weight = float(
            config.get("conditional_branch_loss_weight", 0.0)
        )
        self.branch_aux_score_mode = str(
            config.get("branch_aux_score_mode", "full")
        )
        if self.item_branch_loss_weight < 0.0:
            raise ValueError("item_branch_loss_weight must be >= 0")
        if self.conditional_branch_loss_weight < 0.0:
            raise ValueError("conditional_branch_loss_weight must be >= 0")
        if self.branch_aux_score_mode not in {"full", "incremental"}:
            raise ValueError(
                "branch_aux_score_mode must be 'full' or 'incremental'"
            )
        self.item_aware_mode = str(
            config.get("item_aware_mode", "candidate_delta_residual")
        )
        if self.item_aware_mode not in {
            "orth_gate",
            "orth_gate_confidence_direct",
            "orth_gate_signed_direct",
            "orth_gate_dropped_recovery",
            "orth_gate_bidirectional_correction",
            "orth_gate_layer2_only",
            "orth_gate_normalized_cross",
            "orth_gate_bounded_normalized_cross",
            "orth_gate_projected_bounded_cross",
            "orth_gate_group_cosine",
            "candidate_redecompose",
            "candidate_interest_gate",
            "candidate_interest_residual",
            "candidate_delta_residual",
            "candidate_interaction_residual",
            "candidate_interaction_gate_residual",
            "candidate_interaction_social_residual",
            "candidate_user_social_interaction_residual",
            "candidate_dual_residual",
            "candidate_dual_separate_residual",
        }:
            raise ValueError(
                "item_aware_mode must be 'orth_gate', "
                "'orth_gate_confidence_direct', "
                "'orth_gate_signed_direct', "
                "'orth_gate_dropped_recovery', "
                "'orth_gate_bidirectional_correction', "
                "'orth_gate_layer2_only', "
                "'orth_gate_normalized_cross', "
                "'orth_gate_bounded_normalized_cross', "
                "'orth_gate_projected_bounded_cross', "
                "'orth_gate_group_cosine', "
                "'candidate_redecompose', 'candidate_interest_gate', or "
                "'candidate_interest_residual', or "
                "'candidate_delta_residual', or "
                "'candidate_interaction_residual', or "
                "'candidate_interaction_gate_residual', or "
                "'candidate_interaction_social_residual', or "
                "'candidate_user_social_interaction_residual', or "
                "'candidate_dual_residual', or "
                "'candidate_dual_separate_residual'"
            )

        # coalesce
        self.adj_uu = data_obj.adj_uu.coalesce()
        self.adj_ui = data_obj.adj_ui.coalesce()
        self.sparse_norm_adj = data_obj.norm_adj.coalesce()

        self.layers = nn.ModuleList([
            ODSRLayer(
                self.emb_size,
                self.dropout_val,
                self.candidate_residual_bound,
                self.item_residual_bound,
                self.conditional_residual_bound,
                self.normalized_cross_scale,
                self.item_cosine_groups,
            )
            for _ in range(self.n_layers)
        ])

        if self.item_aware_mode in {
            "candidate_delta_residual",
            "candidate_dual_residual",
            "candidate_dual_separate_residual",
        }:
            # If both branches remained zero-initialized, q-m and all related
            # gradients would be zero. Identity initialization gives
            # r_{u|i}=e_i at step zero, while the zero residual gate still
            # preserves the exact original ODSR scores. User-conditioned
            # cross/cosine corrections are then learned during training.
            for layer in self.layers:
                nn.init.eye_(layer.W_item_anchor.weight)

        self.embedding_dict = nn.ParameterDict({
            "user_emb": nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.user_num, self.emb_size))),
            "item_emb": nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.item_num, self.emb_size))),
        })

        # learnable weighted pooling
        self.alphas = nn.Parameter(torch.ones(self.n_layers + 1))
        self.current_epoch = 0
        self.confidence_warmup_active = False

    def encode(self, current_ssl_mode: Optional[str] = None, return_context: bool = False):
        h_u = self.embedding_dict["user_emb"]
        h_i = self.embedding_dict["item_emb"]

        # social buffer initialized as h_u^(0)
        s_u = h_u.clone()

        h_u_list = [h_u]
        h_i_list = [h_i]
        layer_contexts = []

        for k in range(self.n_layers):
            # LightGCN propagation on expanded bipartite norm_adj
            ego = torch.cat([h_u, h_i], 0)
            side = torch.sparse.mm(self.sparse_norm_adj, ego)
            _, h_i_new = torch.split(side, [self.user_num, self.item_num])

            layer_ssl = "adaptive" if current_ssl_mode == "adaptive" else None
            (
                h_u_new,
                s_clean,
                gate,
                gate_logit,
                base,
                s_orth,
                m_u_item,
                m_u_social,
            ) = self.layers[k](
                h_i=h_i,
                s_u=s_u,
                adj_ui=self.adj_ui,
                adj_uu=self.adj_uu,
                ssl_mode=layer_ssl,
                eps=self.eps,
            )

            if return_context:
                layer_contexts.append({
                    "base": base,
                    "orth": s_orth,
                    "global_gate": gate,
                    "global_gate_logit": gate_logit,
                    "item_interest": m_u_item,
                    "social_message": m_u_social,
                    # Layer k user decomposition was computed from h_i^(k),
                    # so use that same item view for its candidate gate.
                    "candidate_item": h_i,
                })

            # unified social-state update:
            # s^{(l+1)} = s_clean^{(l)}
            s_u = s_clean

            h_u, h_i = h_u_new, h_i_new
            h_u_list.append(h_u)
            h_i_list.append(h_i)

        w = F.softmax(self.alphas, dim=0)
        final_user = torch.sum(torch.stack(h_u_list, dim=1) * w.view(1, -1, 1), dim=1)
        final_item = torch.sum(torch.stack(h_i_list, dim=1) * w.view(1, -1, 1), dim=1)
        if return_context:
            context: Dict[str, Any] = {
                "initial_user": h_u_list[0],
                "layers": layer_contexts,
                "weights": w,
                "final_user": final_user,
                "final_item": final_item,
            }
            return final_user, final_item, context
        return final_user, final_item

    def forward(self, current_ssl_mode: Optional[str] = None):
        return self.encode(current_ssl_mode=current_ssl_mode, return_context=False)

    @staticmethod
    def detach_scoring_context(context: Dict[str, Any]) -> Dict[str, Any]:
        """Detach every ODSR feature used by candidate-aware readout.

        Candidate-head parameters remain live because they are not stored in
        the context.  Consequently a loss scored with this context can train
        the candidate branch without sending gradients into propagation,
        embeddings, the global ODSR gate, or layer-pooling weights.
        """
        return {
            "initial_user": context["initial_user"].detach(),
            "weights": context["weights"].detach(),
            "final_user": context["final_user"].detach(),
            "final_item": context["final_item"].detach(),
            "layers": [
                {
                    name: value.detach()
                    for name, value in layer_context.items()
                }
                for layer_context in context["layers"]
            ],
        }

    def score_dual_separate_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        context: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score fused, raw-item-only, and conditional-only readouts."""
        if self.item_aware_mode != "candidate_dual_separate_residual":
            raise RuntimeError(
                "branch scores require candidate_dual_separate_residual"
            )
        w = context["weights"]
        initial = w[0] * context["initial_user"][users]
        fused_user = initial
        item_user = initial
        conditional_user = initial

        for k, layer_context in enumerate(context["layers"]):
            global_gate = layer_context["global_gate"][users]
            item_interest = layer_context["item_interest"][users]
            s_orth = layer_context["orth"][users]
            candidate = layer_context["candidate_item"][items]
            fused_gate, _, item_delta, conditional_delta = self.layers[
                k
            ].candidate_dual_separate_residual_gate(
                global_gate,
                item_interest,
                s_orth,
                candidate,
            )
            global_logit = torch.logit(
                global_gate.clamp(1e-6, 1.0 - 1e-6)
            )
            item_gate = torch.sigmoid(global_logit + item_delta)
            conditional_gate = torch.sigmoid(
                global_logit + conditional_delta
            )
            base = layer_context["base"][users]
            fused_user = fused_user + w[k + 1] * (
                base + fused_gate * s_orth
            )
            item_user = item_user + w[k + 1] * (
                base + item_gate * s_orth
            )
            conditional_user = conditional_user + w[k + 1] * (
                base + conditional_gate * s_orth
            )

        item_e = context["final_item"][items]
        return (
            (fused_user * item_e).sum(dim=-1),
            (item_user * item_e).sum(dim=-1),
            (conditional_user * item_e).sum(dim=-1),
        )

    def score_dual_separate_increment_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        context: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score only each branch's change from the global ODSR readout."""
        if self.item_aware_mode != "candidate_dual_separate_residual":
            raise RuntimeError(
                "increment scores require candidate_dual_separate_residual"
            )
        w = context["weights"]
        initial = context["initial_user"][users]
        item_increment_user = torch.zeros_like(initial)
        conditional_increment_user = torch.zeros_like(initial)

        for k, layer_context in enumerate(context["layers"]):
            global_gate = layer_context["global_gate"][users]
            item_interest = layer_context["item_interest"][users]
            s_orth = layer_context["orth"][users]
            candidate = layer_context["candidate_item"][items]
            _, _, item_delta, conditional_delta = self.layers[
                k
            ].candidate_dual_separate_residual_gate(
                global_gate,
                item_interest,
                s_orth,
                candidate,
            )
            global_logit = torch.logit(
                global_gate.clamp(1e-6, 1.0 - 1e-6)
            )
            item_gate = torch.sigmoid(global_logit + item_delta)
            conditional_gate = torch.sigmoid(
                global_logit + conditional_delta
            )
            item_increment_user = item_increment_user + w[k + 1] * (
                (item_gate - global_gate) * s_orth
            )
            conditional_increment_user = (
                conditional_increment_user
                + w[k + 1]
                * (conditional_gate - global_gate)
                * s_orth
            )

        item_e = context["final_item"][items]
        return (
            (item_increment_user * item_e).sum(dim=-1),
            (conditional_increment_user * item_e).sum(dim=-1),
        )

    def score_confidence_proposal_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Score with proposal gates while detaching every shared feature.

        This auxiliary path trains only ``W_item_gate``,
        ``item_cross_scale``, and ``item_cos_scale``. It cannot update the
        embeddings, ODSR propagation/gate parameters, or pooling weights.
        """
        if self.item_aware_mode != "orth_gate_confidence_direct":
            raise RuntimeError(
                "proposal scores require orth_gate_confidence_direct"
            )
        weights = context["weights"].detach()
        pair_user = (
            weights[0] * context["initial_user"][users].detach()
        )
        for layer_index, layer_context in enumerate(context["layers"]):
            base = layer_context["base"][users].detach()
            s_orth = layer_context["orth"][users].detach()
            candidate = layer_context["candidate_item"][items].detach()
            proposal_raw = self.layers[layer_index].item_aware_gate_delta(
                s_orth,
                candidate,
            )
            proposal = torch.sigmoid(proposal_raw)
            pair_user = pair_user + weights[layer_index + 1] * (
                base + proposal * s_orth
            )
        item_embedding = context["final_item"][items].detach()
        return (pair_user * item_embedding).sum(dim=-1)

    def signed_gate_change_penalty(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Mean actual gate change, updating only candidate-gate parameters."""
        penalties = []
        for layer_index, layer_context in enumerate(context["layers"]):
            global_gate = layer_context["global_gate"][users].detach()
            s_orth = layer_context["orth"][users].detach()
            candidate = layer_context["candidate_item"][items].detach()
            _, gate_change, _, _ = self.layers[
                layer_index
            ].item_aware_signed_direct_gate(
                global_gate,
                s_orth,
                candidate,
                self.signed_gate_scale,
                return_parts=True,
            )
            penalties.append(gate_change.abs().mean())
        return torch.stack(penalties).mean()

    def score_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        context: Dict[str, Any],
        return_pair_user: bool = False,
        detach_shared_parameters: bool = False,
    ):
        """Score aligned (user, item) pairs without another graph pass."""
        w = context["weights"]
        pair_user = w[0] * context["initial_user"][users]
        for k, layer_context in enumerate(context["layers"]):
            global_gate = layer_context["global_gate"][users]
            candidate = layer_context["candidate_item"][items]
            if self.item_aware_mode == "candidate_dual_separate_residual":
                item_interest = layer_context["item_interest"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _, _, _ = self.layers[
                    k
                ].candidate_dual_separate_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif self.item_aware_mode == "candidate_dual_residual":
                item_interest = layer_context["item_interest"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _, _ = self.layers[k].candidate_dual_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif self.item_aware_mode == "candidate_delta_residual":
                item_interest = layer_context["item_interest"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _ = self.layers[k].candidate_delta_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif self.item_aware_mode == "candidate_interaction_residual":
                item_interest = layer_context["item_interest"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _, _ = self.layers[
                    k
                ].candidate_interaction_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif self.item_aware_mode == "candidate_interaction_gate_residual":
                item_interest = layer_context["item_interest"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _ = self.layers[
                    k
                ].candidate_interaction_gate_residual(
                    global_gate,
                    item_interest,
                    candidate,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif self.item_aware_mode == "candidate_interaction_social_residual":
                item_interest = layer_context["item_interest"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _, _ = self.layers[
                    k
                ].candidate_interaction_social_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif (
                self.item_aware_mode
                == "candidate_user_social_interaction_residual"
            ):
                item_interest = layer_context["item_interest"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _, _, _ = self.layers[
                    k
                ].candidate_user_social_interaction_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif self.item_aware_mode == "candidate_interest_residual":
                item_interest = layer_context["item_interest"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _ = self.layers[k].candidate_interest_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif self.item_aware_mode == "candidate_interest_gate":
                item_interest = layer_context["item_interest"][users]
                social_message = layer_context["social_message"][users]
                s_orth = layer_context["orth"][users]
                gate_ui, _ = self.layers[k].candidate_interest_gate(
                    item_interest,
                    social_message,
                    s_orth,
                    candidate,
                    detach_odsr_gate=detach_shared_parameters,
                )
                layer_user = layer_context["base"][users] + gate_ui * s_orth
            elif self.item_aware_mode == "candidate_redecompose":
                item_interest = layer_context["item_interest"][users]
                social_message = layer_context["social_message"][users]
                useful_social, _, _ = self.layers[k].candidate_redecompose_social(
                    item_interest,
                    social_message,
                    global_gate,
                    candidate,
                )
                layer_user = item_interest + useful_social
            else:
                base = layer_context["base"][users]
                s_orth = layer_context["orth"][users]
                if (
                    self.item_aware_mode == "orth_gate_layer2_only"
                    and k != 1
                ):
                    gate_ui = global_gate
                elif self.item_aware_mode == "orth_gate_confidence_direct":
                    if self.confidence_warmup_active:
                        gate_ui = global_gate
                    else:
                        gate_ui = self.layers[
                            k
                        ].item_aware_confidence_direct_gate(
                            global_gate,
                            s_orth,
                            candidate,
                            self.candidate_confidence_bias,
                        )
                elif self.item_aware_mode == "orth_gate_signed_direct":
                    gate_ui = self.layers[k].item_aware_signed_direct_gate(
                        global_gate,
                        s_orth,
                        candidate,
                        self.signed_gate_scale,
                    )
                elif self.item_aware_mode == "orth_gate_dropped_recovery":
                    gate_ui = self.layers[
                        k
                    ].item_aware_dropped_recovery_gate(
                        global_gate,
                        s_orth,
                        candidate,
                        self.candidate_recovery_scale,
                    )
                elif (
                    self.item_aware_mode
                    == "orth_gate_bidirectional_correction"
                ):
                    gate_ui = self.layers[k].item_aware_bidirectional_gate(
                        global_gate,
                        s_orth,
                        candidate,
                        self.candidate_recovery_scale,
                    )
                elif self.item_aware_mode in {
                    "orth_gate_normalized_cross",
                    "orth_gate_bounded_normalized_cross",
                }:
                    gate_ui = self.layers[
                        k
                    ].item_aware_gate_normalized_cross(
                        global_gate,
                        s_orth,
                        candidate,
                        bound_cross=(
                            self.item_aware_mode
                            == "orth_gate_bounded_normalized_cross"
                        ),
                    )
                elif (
                    self.item_aware_mode
                    == "orth_gate_projected_bounded_cross"
                ):
                    gate_ui = self.layers[k].item_aware_gate_projected_cross(
                        global_gate, s_orth, candidate
                    )
                elif self.item_aware_mode == "orth_gate_group_cosine":
                    gate_ui = self.layers[k].item_aware_gate_group_cosine(
                        global_gate, s_orth, candidate
                    )
                else:
                    gate_ui = self.layers[k].item_aware_gate(
                        global_gate, s_orth, candidate
                    )
                layer_user = base + gate_ui * s_orth
            pair_user = pair_user + w[k + 1] * layer_user

        item_e = context["final_item"][items]
        scores = (pair_user * item_e).sum(dim=-1)
        if return_pair_user:
            return scores, pair_user, item_e
        return scores

    def score_item_chunk(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Score the Cartesian product of users and one candidate item chunk."""
        w = context["weights"]
        pair_user = w[0] * context["initial_user"][users].unsqueeze(1)

        for k, layer_context in enumerate(context["layers"]):
            global_gate = layer_context["global_gate"][users].unsqueeze(1)
            candidate = layer_context["candidate_item"][items].unsqueeze(0)
            if self.item_aware_mode == "candidate_dual_separate_residual":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _, _, _ = self.layers[
                    k
                ].candidate_dual_separate_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif self.item_aware_mode == "candidate_dual_residual":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _, _ = self.layers[k].candidate_dual_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif self.item_aware_mode == "candidate_delta_residual":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _ = self.layers[k].candidate_delta_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif self.item_aware_mode == "candidate_interaction_residual":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _, _ = self.layers[
                    k
                ].candidate_interaction_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif self.item_aware_mode == "candidate_interaction_gate_residual":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _ = self.layers[
                    k
                ].candidate_interaction_gate_residual(
                    global_gate,
                    item_interest,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif self.item_aware_mode == "candidate_interaction_social_residual":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _, _ = self.layers[
                    k
                ].candidate_interaction_social_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif (
                self.item_aware_mode
                == "candidate_user_social_interaction_residual"
            ):
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _, _, _ = self.layers[
                    k
                ].candidate_user_social_interaction_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif self.item_aware_mode == "candidate_interest_residual":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _ = self.layers[k].candidate_interest_residual_gate(
                    global_gate,
                    item_interest,
                    s_orth,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif self.item_aware_mode == "candidate_interest_gate":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                social_message = layer_context["social_message"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                gate_ui, _ = self.layers[k].candidate_interest_gate(
                    item_interest,
                    social_message,
                    s_orth,
                    candidate,
                )
                layer_user = (
                    layer_context["base"][users].unsqueeze(1)
                    + gate_ui * s_orth
                )
            elif self.item_aware_mode == "candidate_redecompose":
                item_interest = layer_context["item_interest"][users].unsqueeze(1)
                social_message = layer_context["social_message"][users].unsqueeze(1)
                useful_social, _, _ = self.layers[k].candidate_redecompose_social(
                    item_interest,
                    social_message,
                    global_gate,
                    candidate,
                )
                layer_user = item_interest + useful_social
            else:
                base = layer_context["base"][users].unsqueeze(1)
                s_orth = layer_context["orth"][users].unsqueeze(1)
                if (
                    self.item_aware_mode == "orth_gate_layer2_only"
                    and k != 1
                ):
                    gate_ui = global_gate
                elif self.item_aware_mode == "orth_gate_confidence_direct":
                    if self.confidence_warmup_active:
                        gate_ui = global_gate
                    else:
                        gate_ui = self.layers[
                            k
                        ].item_aware_confidence_direct_gate(
                            global_gate,
                            s_orth,
                            candidate,
                            self.candidate_confidence_bias,
                        )
                elif self.item_aware_mode == "orth_gate_signed_direct":
                    gate_ui = self.layers[k].item_aware_signed_direct_gate(
                        global_gate,
                        s_orth,
                        candidate,
                        self.signed_gate_scale,
                    )
                elif self.item_aware_mode == "orth_gate_dropped_recovery":
                    gate_ui = self.layers[
                        k
                    ].item_aware_dropped_recovery_gate(
                        global_gate,
                        s_orth,
                        candidate,
                        self.candidate_recovery_scale,
                    )
                elif (
                    self.item_aware_mode
                    == "orth_gate_bidirectional_correction"
                ):
                    gate_ui = self.layers[k].item_aware_bidirectional_gate(
                        global_gate,
                        s_orth,
                        candidate,
                        self.candidate_recovery_scale,
                    )
                elif self.item_aware_mode in {
                    "orth_gate_normalized_cross",
                    "orth_gate_bounded_normalized_cross",
                }:
                    gate_ui = self.layers[
                        k
                    ].item_aware_gate_normalized_cross(
                        global_gate,
                        s_orth,
                        candidate,
                        bound_cross=(
                            self.item_aware_mode
                            == "orth_gate_bounded_normalized_cross"
                        ),
                    )
                elif (
                    self.item_aware_mode
                    == "orth_gate_projected_bounded_cross"
                ):
                    gate_ui = self.layers[k].item_aware_gate_projected_cross(
                        global_gate, s_orth, candidate
                    )
                elif self.item_aware_mode == "orth_gate_group_cosine":
                    gate_ui = self.layers[k].item_aware_gate_group_cosine(
                        global_gate, s_orth, candidate
                    )
                else:
                    gate_ui = self.layers[k].item_aware_gate(
                        global_gate, s_orth, candidate
                    )
                layer_user = base + gate_ui * s_orth
            pair_user = pair_user + w[k + 1] * layer_user

        item_e = context["final_item"][items]
        return (pair_user * item_e.unsqueeze(0)).sum(dim=-1)

    def score_all_items(
        self,
        users: torch.Tensor,
        context: Dict[str, Any],
        item_batch: int = 512,
    ) -> torch.Tensor:
        """Full-ranking scores, chunked to bound [B, items, D] memory."""
        chunks = []
        for start in range(0, self.item_num, item_batch):
            items = torch.arange(
                start,
                min(start + item_batch, self.item_num),
                device=users.device,
            )
            chunks.append(self.score_item_chunk(users, items, context))
        return torch.cat(chunks, dim=1)

    @torch.no_grad()
    def mine_hard_negatives(
        self,
        users: torch.Tensor,
        candidates: torch.Tensor,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Return the Top-K model-scored items from each unseen candidate pool."""
        if candidates.dim() != 2:
            raise ValueError("hard-negative candidates must have shape [B, pool]")
        if self.negative_k < 1 or self.negative_k > candidates.size(1):
            raise ValueError(
                "negative_k must be between 1 and hard candidate count "
                f"(got negative_k={self.negative_k}, "
                f"candidates={candidates.size(1)})"
            )
        candidate_users = users.unsqueeze(1).expand_as(candidates).reshape(-1)
        candidate_items = candidates.reshape(-1)
        candidate_score = self.score_pairs(
            candidate_users,
            candidate_items,
            context,
        ).view_as(candidates)
        hard_index = torch.topk(
            candidate_score,
            k=self.negative_k,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        return candidates.gather(1, hard_index)

    def compute_loss(
        self,
        users: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        backbone_neg: Optional[torch.Tensor] = None,
    ):
        final_user, final_item, context = self.encode(current_ssl_mode=None, return_context=True)
        u_e = final_user[users]
        p_e = final_item[pos]
        if neg.dim() == 1:
            neg = neg.unsqueeze(1)

        isolate_candidate_negatives = (
            self.negative_gradient_mode == "candidate_only"
        )
        if isolate_candidate_negatives:
            if backbone_neg is None:
                raise ValueError(
                    "candidate_only mode requires an independently sampled "
                    "backbone_neg"
                )
            if backbone_neg.dim() == 1:
                backbone_neg = backbone_neg.unsqueeze(1)
            if backbone_neg.shape != (users.numel(), 1):
                raise ValueError(
                    "backbone_neg must have shape [batch, 1] in "
                    "candidate_only mode"
                )

        if self.negative_strategy == "hard_topk_mean":
            # Mine with the actual V1 candidate-aware score, but do not keep
            # gradients for all candidates. Only the selected Top-K negatives
            # are scored again below and participate in the BPR update.
            neg = self.mine_hard_negatives(users, neg, context)

        # Regularization is part of the backbone objective.  In isolated mode
        # it must see only the K=1 ODSR negative, never the candidate K=8 set.
        regularization_neg = (
            backbone_neg if isolate_candidate_negatives else neg
        )
        n_e = final_item[regularization_neg]

        neg_users = users.unsqueeze(1).expand_as(neg).reshape(-1)
        neg_items = neg.reshape(-1)
        use_confidence_proposal_loss = (
            self.item_aware_mode == "orth_gate_confidence_direct"
            and (
                self.proposal_warmup_loss_weight > 0.0
                or self.proposal_aux_loss_weight > 0.0
            )
        )
        use_branch_losses = (
            self.item_aware_mode == "candidate_dual_separate_residual"
            and (
                self.item_branch_loss_weight > 0.0
                or self.conditional_branch_loss_weight > 0.0
            )
        )
        if isolate_candidate_negatives:
            # Preserve the original candidate-independent ODSR update with a
            # single negative.  Do not call score_pairs here: it contains the
            # candidate-aware residual gate.
            base_pos_score = (u_e * p_e).sum(dim=-1)
            base_neg_score = (
                u_e.unsqueeze(1) * final_item[backbone_neg]
            ).sum(dim=-1)
            backbone_rec_loss = multi_negative_bpr_loss(
                base_pos_score,
                base_neg_score,
                "random_mean",
            )

            # K negatives supervise the candidate readout, but every feature
            # originating in ODSR is stop-gradient.  For orth_gate this leaves
            # W_item_gate/item_cross_scale/item_cos_scale as the only trainable
            # path from this loss.
            candidate_context = self.detach_scoring_context(context)
            candidate_pos_score = self.score_pairs(
                users,
                pos,
                candidate_context,
                detach_shared_parameters=True,
            )
            candidate_neg_score = self.score_pairs(
                neg_users,
                neg_items,
                candidate_context,
                detach_shared_parameters=True,
            ).view_as(neg)
            candidate_rec_loss = multi_negative_bpr_loss(
                candidate_pos_score,
                candidate_neg_score,
                self.negative_strategy,
            )

            # Deliberately do not divide by (1 + candidate_loss_weight): that
            # would rescale the original ODSR K=1 gradient.
            rec_loss = (
                backbone_rec_loss
                + self.candidate_loss_weight * candidate_rec_loss
            )
            self.last_rec_components = {
                "backbone_k1": backbone_rec_loss.detach(),
                "candidate": candidate_rec_loss.detach(),
                "candidate_weight": torch.tensor(
                    self.candidate_loss_weight,
                    device=backbone_rec_loss.device,
                ),
            }
        elif use_confidence_proposal_loss:
            pos_score = self.score_pairs(users, pos, context)
            neg_score = self.score_pairs(
                neg_users, neg_items, context
            ).view_as(neg)
            pos_proposal_score = self.score_confidence_proposal_pairs(
                users,
                pos,
                context,
            )
            neg_proposal_score = self.score_confidence_proposal_pairs(
                neg_users,
                neg_items,
                context,
            ).view_as(neg)
            fused_rec_loss = multi_negative_bpr_loss(
                pos_score,
                neg_score,
                self.negative_strategy,
            )
            proposal_rec_loss = multi_negative_bpr_loss(
                pos_proposal_score,
                neg_proposal_score,
                self.negative_strategy,
            )
            if self.current_epoch <= self.confidence_warmup_epochs:
                proposal_weight = self.proposal_warmup_loss_weight
            else:
                proposal_weight = self.proposal_aux_loss_weight
            # The proposal path is fully detached from the shared backbone,
            # so keep the original fused-BPR gradient scale exactly intact.
            rec_loss = fused_rec_loss + proposal_weight * proposal_rec_loss
            self.last_rec_components = {
                "fused": fused_rec_loss.detach(),
                "proposal": proposal_rec_loss.detach(),
                "proposal_weight": torch.tensor(
                    proposal_weight,
                    device=fused_rec_loss.device,
                ),
            }
        elif use_branch_losses:
            if self.branch_aux_score_mode == "incremental":
                pos_score = self.score_pairs(users, pos, context)
                neg_score = self.score_pairs(
                    neg_users, neg_items, context
                ).view_as(neg)
                pos_item_score, pos_conditional_score = (
                    self.score_dual_separate_increment_pairs(
                        users, pos, context
                    )
                )
                neg_item_score, neg_conditional_score = (
                    self.score_dual_separate_increment_pairs(
                        neg_users, neg_items, context
                    )
                )
            else:
                pos_score, pos_item_full_score, pos_conditional_full_score = (
                    self.score_dual_separate_pairs(users, pos, context)
                )
                neg_score, neg_item_full_score, neg_conditional_full_score = (
                    self.score_dual_separate_pairs(
                        neg_users,
                        neg_items,
                        context,
                    )
                )
                neg_score = neg_score.view_as(neg)
                pos_item_score = pos_item_full_score
                pos_conditional_score = pos_conditional_full_score
                neg_item_score = neg_item_full_score
                neg_conditional_score = neg_conditional_full_score
            neg_item_score = neg_item_score.view_as(neg)
            neg_conditional_score = neg_conditional_score.view_as(neg)
            fused_rec_loss = multi_negative_bpr_loss(
                pos_score, neg_score, self.negative_strategy
            )
            item_rec_loss = multi_negative_bpr_loss(
                pos_item_score, neg_item_score, self.negative_strategy
            )
            conditional_rec_loss = multi_negative_bpr_loss(
                pos_conditional_score,
                neg_conditional_score,
                self.negative_strategy,
            )
            rec_weight_sum = (
                1.0
                + self.item_branch_loss_weight
                + self.conditional_branch_loss_weight
            )
            rec_loss = (
                fused_rec_loss
                + self.item_branch_loss_weight * item_rec_loss
                + self.conditional_branch_loss_weight * conditional_rec_loss
            ) / rec_weight_sum
            self.last_rec_components = {
                "fused": fused_rec_loss.detach(),
                "item": item_rec_loss.detach(),
                "conditional": conditional_rec_loss.detach(),
            }
        else:
            pos_score = self.score_pairs(users, pos, context)
            neg_score = self.score_pairs(
                neg_users, neg_items, context
            ).view_as(neg)
            rec_loss = multi_negative_bpr_loss(
                pos_score,
                neg_score,
                self.negative_strategy,
            )
            self.last_rec_components = {"fused": rec_loss.detach()}

        gate_change_penalty = torch.tensor(0.0, device=self.device)
        if (
            self.item_aware_mode == "orth_gate_signed_direct"
            and self.gate_change_reg_weight > 0.0
        ):
            positive_penalty = self.signed_gate_change_penalty(
                users,
                pos,
                context,
            )
            negative_penalty = self.signed_gate_change_penalty(
                neg_users,
                neg_items,
                context,
            )
            gate_change_penalty = 0.5 * (
                positive_penalty + negative_penalty
            )
            self.last_rec_components["gate_change"] = (
                gate_change_penalty.detach()
            )
        gate_saturation_penalty = torch.tensor(0.0, device=self.device)
        if self.gate_saturation_reg_weight > 0.0:
            layer_penalties = []
            for layer_context in context["layers"]:
                gate_logit = layer_context["global_gate_logit"][users]
                excess = F.relu(
                    gate_logit.abs() - self.gate_saturation_bound
                )
                layer_penalties.append(excess.square().mean())
            gate_saturation_penalty = torch.stack(layer_penalties).mean()
            self.last_rec_components["gate_saturation"] = (
                gate_saturation_penalty.detach()
            )
        reg_loss = self.reg * (u_e.norm(2) + p_e.norm(2) + n_e.norm(2))

        ssl_loss = torch.tensor(0.0, device=self.device)
        if self.ssl_reg > 0:
            # adaptive view vs clean view
            v1_u, _ = self.forward(current_ssl_mode="adaptive")
            v2_u, _ = self.forward(current_ssl_mode=None)

            v1 = F.normalize(v1_u[users], dim=1)
            v2 = F.normalize(v2_u[users], dim=1)

            pos_sc = torch.exp((v1 * v2).sum(1) / self.ssl_temp)
            ttl_sc = torch.matmul(v1, v2.t()).div(self.ssl_temp).exp().sum(1)
            ssl_loss = self.ssl_reg * (-torch.log(pos_sc / ttl_sc).mean())

        return (
            rec_loss
            + self.gate_change_reg_weight * gate_change_penalty
            + self.gate_saturation_reg_weight * gate_saturation_penalty
            + reg_loss
            + ssl_loss
        )
