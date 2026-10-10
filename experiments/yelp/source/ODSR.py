"""ODSR model plus a candidate-conditioned social evidence retriever.

The historical implementation is kept in a frozen Python 3.9 bytecode module so
existing ODSR and candidate modes remain bit-for-bit compatible.  The subclass
below adds the retrieval expert without changing the ODSR backbone.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


import ODSR_base as _base

ODSRLayer = _base.ODSRLayer


class ODSR(_base.ODSR):
    """Backward-compatible ODSR with an optional social retrieval expert.

    The expert constructs multiple interests from each user's training history,
    propagates those interests only over non-self social edges, and lets a
    candidate retrieve useful social evidence.  A learned null choice allows
    exact abstention when the retrieved social evidence is weak or inconsistent
    with the user's own candidate-local interest.
    """

    def __init__(
        self,
        data_obj,
        config: Dict,
        device: torch.device,
        ablation_flags: Optional[Dict] = None,
    ):
        prototype_enabled = bool(config.get("candidate_retrieval_expert", False))
        direct_enabled = bool(config.get("candidate_direct_history_expert", False))
        if prototype_enabled and direct_enabled:
            raise ValueError(
                "prototype and direct-history experts are mutually exclusive"
            )
        retrieval_enabled = prototype_enabled or direct_enabled
        if not retrieval_enabled:
            super().__init__(data_obj, config, device, ablation_flags)
            self.candidate_retrieval_expert = False
            self.candidate_direct_history_expert = False
            return

        # Initialize a pure ODSR backbone.  Candidate scoring is supplied by the
        # new expert below, rather than by any historical residual branch.
        base_config = dict(config)
        for key in (
            "candidate_subspace",
            "candidate_dimensionwise",
            "candidate_lowrank_gate",
            "candidate_history_gate",
            "candidate_noise_filter",
            "candidate_intent_expert",
            "candidate_social_expert",
            "candidate_trust_expert",
        ):
            base_config[key] = False
        super().__init__(data_obj, base_config, device, ablation_flags)

        self.candidate_subspace = True
        self.candidate_retrieval_expert = prototype_enabled
        self.candidate_direct_history_expert = direct_enabled
        self.candidate_dimensionwise = False
        self.candidate_lowrank_gate = False
        self.candidate_history_gate = False
        self.candidate_noise_filter = False
        self.candidate_intent_expert = False
        self.candidate_social_expert = False
        self.candidate_trust_expert = False
        self.candidate_raw_loss = False

        self.candidate_loss_weight = float(config.get("candidate_loss_weight", 0.2))
        self.hard_pool_size = int(config.get("hard_pool_size", 64))
        self.hard_topk = int(config.get("hard_topk", 8))
        self.candidate_item_chunk = int(config.get("candidate_item_chunk", 2048))

        if direct_enabled:
            self._init_direct_history(data_obj, config, device)
            return

        self.retrieval_num_interests = int(
            config.get("candidate_retrieval_interests", 4)
        )
        self.retrieval_tau = float(config.get("candidate_retrieval_tau", 0.2))
        self.retrieval_assignment_tau = float(
            config.get("candidate_retrieval_assignment_tau", 0.5)
        )
        self.retrieval_consistency_weight = float(
            config.get("candidate_retrieval_consistency_weight", 1.0)
        )
        self.retrieval_evidence_threshold = float(
            config.get("candidate_retrieval_threshold", 0.5)
        )
        self.retrieval_residual_max = float(
            config.get("candidate_retrieval_residual_max", 0.25)
        )
        self.retrieval_scale_max = float(
            config.get("candidate_retrieval_scale_max", 0.5)
        )
        self.retrieval_init_scale = float(
            config.get("candidate_retrieval_init_scale", 0.05)
        )
        self.retrieval_reg_weight = float(
            config.get("candidate_retrieval_reg", 1e-4)
        )
        self.retrieval_balance_weight = float(
            config.get("candidate_retrieval_balance_reg", 1e-3)
        )
        self.retrieval_specialization_weight = float(
            config.get("candidate_retrieval_specialization_reg", 1e-3)
        )
        self.retrieval_confidence_weight = float(
            config.get("candidate_retrieval_confidence_weight", 0.05)
        )
        self.retrieval_specificity = bool(
            config.get("candidate_retrieval_specificity", False)
        )
        self.retrieval_dispersion_weight = float(
            config.get("candidate_retrieval_dispersion_weight", 2.0)
        )
        self.retrieval_mass_margin = float(
            config.get("candidate_retrieval_mass_margin", 0.2)
        )
        self.retrieval_mass_margin_weight = float(
            config.get("candidate_retrieval_mass_margin_weight", 0.2)
        )

        if self.retrieval_num_interests < 2:
            raise ValueError("candidate_retrieval_interests must be at least 2")
        if self.retrieval_tau <= 0 or self.retrieval_assignment_tau <= 0:
            raise ValueError("retrieval temperatures must be positive")
        if self.retrieval_consistency_weight < 0:
            raise ValueError("candidate_retrieval_consistency_weight must be nonnegative")
        if self.retrieval_residual_max < 0:
            raise ValueError("candidate_retrieval_residual_max must be nonnegative")
        if not 0 < self.retrieval_init_scale < self.retrieval_scale_max:
            raise ValueError("retrieval init scale must lie strictly between 0 and max")
        if min(
            self.retrieval_reg_weight,
            self.retrieval_balance_weight,
            self.retrieval_specialization_weight,
            self.retrieval_confidence_weight,
            self.retrieval_dispersion_weight,
            self.retrieval_mass_margin,
            self.retrieval_mass_margin_weight,
        ) < 0:
            raise ValueError("retrieval regularization weights must be nonnegative")

        k = self.retrieval_num_interests
        d = self.emb_size
        self.retrieval_item_head = nn.Linear(d, k)
        self.retrieval_query_proj = nn.Linear(d, d, bias=False)
        self.retrieval_proto_proj = nn.Linear(d, d, bias=False)
        self.retrieval_local_proj = nn.Linear(d, d, bias=False)
        self.retrieval_evidence_mlp = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.retrieval_evidence_mlp[-1].weight)
        nn.init.zeros_(self.retrieval_evidence_mlp[-1].bias)
        self.retrieval_gate_mlp = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, d)
        )
        nn.init.zeros_(self.retrieval_gate_mlp[-1].weight)
        nn.init.constant_(self.retrieval_gate_mlp[-1].bias, -0.5)
        scale_ratio = self.retrieval_init_scale / self.retrieval_scale_max
        self.retrieval_scale_logit = nn.Parameter(
            torch.tensor(math.log(scale_ratio / (1.0 - scale_ratio)), device=device)
        )

        # Social retrieval must not leak the user's own history through graph
        # self-loops.  Normalize the remaining outgoing neighbor weights.
        adj = self.adj_uu.coalesce()
        indices = adj.indices()
        values = adj.values()
        nonself = indices[0] != indices[1]
        neighbor_indices = indices[:, nonself]
        neighbor_values = values[nonself]
        row_mass = torch.zeros(self.user_num, device=device, dtype=neighbor_values.dtype)
        if neighbor_values.numel() > 0:
            row_mass.scatter_add_(0, neighbor_indices[0], neighbor_values.abs())
            neighbor_values = neighbor_values / row_mass[
                neighbor_indices[0]
            ].clamp_min(1e-12)
        self.adj_uu_neighbors = torch.sparse_coo_tensor(
            neighbor_indices,
            neighbor_values,
            (self.user_num, self.user_num),
            device=device,
        ).coalesce()

    def _init_direct_history(self, data_obj, config: Dict, device: torch.device):
        """Build fixed-size indices over raw training histories and social links."""
        self.direct_v1_only = bool(
            config.get("candidate_direct_v1_only", False)
        )
        self.direct_user_history_max = int(
            config.get("candidate_direct_user_history_max", 64)
        )
        self.direct_friend_select = int(
            config.get(
                "candidate_direct_friend_select",
                min(4, int(config.get("candidate_direct_friend_max", 8))),
            )
        )
        self.direct_friend_screen_topl = int(
            config.get("candidate_direct_friend_screen_topl", 2)
        )
        self.direct_friend_screen_chunk = int(
            config.get("candidate_direct_friend_screen_chunk", 8)
        )
        self.direct_friend_max = int(
            config.get("candidate_direct_friend_max", 8)
        )
        self.direct_friend_history_max = int(
            config.get("candidate_direct_friend_history_max", 16)
        )
        self.direct_user_topl = int(
            config.get("candidate_direct_user_topl", 8)
        )
        self.direct_eval_topm = int(
            config.get("candidate_direct_eval_topm", 512)
        )
        self.direct_friend_topl = int(
            config.get("candidate_direct_friend_topl", 4)
        )
        self.direct_two_stage_retrieval = bool(
            config.get("candidate_direct_two_stage_retrieval", False)
        )
        self.direct_user_consistent_screen = bool(
            config.get("candidate_direct_user_consistent_screen", False)
        )
        self.direct_screen_consistency_weight = float(
            config.get("candidate_direct_screen_consistency_weight", 0.5)
        )
        self.direct_screen_prior_weight = float(
            config.get("candidate_direct_screen_prior_weight", 0.1)
        )
        self.direct_match_threshold = float(
            config.get("candidate_direct_match_threshold", 0.2)
        )
        self.direct_freeze_epochs = int(
            config.get("candidate_direct_freeze_epochs", 3)
        )
        self.direct_retrieval_lr_scale = float(
            config.get("candidate_direct_retrieval_lr_scale", 0.1)
        )
        if self.direct_v1_only:
            # These controls were introduced after direct-history v1.  The
            # original v1 trained its retrieval projections from epoch one at
            # the optimizer learning rate.
            self.direct_freeze_epochs = 0
            self.direct_retrieval_lr_scale = 1.0
        self.direct_tau = float(config.get("candidate_direct_tau", 0.2))
        self.direct_consistency_weight = float(
            config.get("candidate_direct_consistency_weight", 1.0)
        )
        self.direct_evidence_threshold = float(
            config.get("candidate_direct_threshold", 0.5)
        )
        self.direct_residual_max = float(
            config.get("candidate_direct_residual_max", 0.25)
        )
        self.direct_scale_max = float(
            config.get("candidate_direct_scale_max", 0.5)
        )
        self.direct_init_scale = float(
            config.get("candidate_direct_init_scale", 0.05)
        )
        self.direct_reg_weight = float(
            config.get("candidate_direct_reg", 1e-4)
        )
        self.direct_confidence_weight = float(
            config.get("candidate_direct_confidence_weight", 0.05)
        )
        self.direct_rank_calibrated = bool(
            config.get("candidate_direct_rank_calibrated", False)
        )
        self.direct_novelty_v1 = bool(
            config.get("candidate_direct_novelty_v1", False)
        )
        self.direct_target_margin_ratio = float(
            config.get("candidate_direct_target_margin_ratio", 0.05)
        )
        self.direct_max_margin_ratio = float(
            config.get("candidate_direct_max_margin_ratio", 0.20)
        )
        self.direct_rank_beta = float(
            config.get("candidate_direct_rank_beta", 0.05)
        )
        self.direct_correction_reg_weight = float(
            config.get("candidate_direct_correction_reg", 0.05)
        )
        self.direct_score_scale_floor = float(
            config.get("candidate_direct_score_scale_floor", 0.05)
        )
        self.direct_router_interests = int(
            config.get("candidate_direct_router_interests", 4)
        )
        self.direct_router_tau = float(
            config.get("candidate_direct_router_tau", 0.5)
        )
        self.direct_router_weight = float(
            config.get("candidate_direct_router_weight", 0.2)
        )
        self.direct_router_balance_weight = float(
            config.get("candidate_direct_router_balance_reg", 1e-3)
        )
        self.direct_router_entropy_weight = float(
            config.get("candidate_direct_router_entropy_reg", 1e-3)
        )
        if self.direct_v1_only:
            self.direct_router_interests = 1
            self.direct_router_weight = 0.0
            self.direct_router_balance_weight = 0.0
            self.direct_router_entropy_weight = 0.0
        self.direct_calibrated_trust = bool(
            config.get("candidate_direct_calibrated_trust", False)
        )
        self.direct_calibration_warmup = int(
            config.get("candidate_direct_calibration_warmup", 3)
        )
        self.direct_calibration_interval = int(
            config.get("candidate_direct_calibration_interval", 5)
        )
        self.direct_calibration_items = int(
            config.get("candidate_direct_calibration_items", 8)
        )
        self.direct_background_items = int(
            config.get("candidate_direct_background_items", 32)
        )
        self.direct_trust_tau = float(
            config.get("candidate_direct_trust_tau", 0.2)
        )
        self.direct_trust_shrinkage = float(
            config.get("candidate_direct_trust_shrinkage", 5.0)
        )
        self.direct_lift_margin = float(
            config.get("candidate_direct_lift_margin", 0.0)
        )
        self.direct_prior_weight = float(
            config.get("candidate_direct_prior_weight", 0.1)
        )
        self.direct_hierarchical_prototypes = bool(
            config.get("candidate_direct_hierarchical_prototypes", False)
        )
        self.direct_fine_interests = int(
            config.get("candidate_direct_fine_interests", 8)
        )
        self.direct_coarse_interests = int(
            config.get("candidate_direct_coarse_interests", 4)
        )
        self.direct_fine_topk = int(
            config.get("candidate_direct_fine_topk", 2)
        )
        self.direct_prototype_warmup = int(
            config.get("candidate_direct_prototype_warmup", 3)
        )
        self.direct_prototype_interval = int(
            config.get("candidate_direct_prototype_interval", 5)
        )
        self.direct_kmeans_iters = int(
            config.get("candidate_direct_kmeans_iters", 8)
        )
        if min(
            self.direct_user_history_max,
            self.direct_friend_max,
            self.direct_friend_select,
            self.direct_friend_screen_topl,
            self.direct_friend_screen_chunk,
            self.direct_friend_history_max,
            self.direct_user_topl,
            self.direct_eval_topm,
            self.direct_friend_topl,
        ) <= 0:
            raise ValueError("direct-history sizes and Top-L values must be positive")
        if self.direct_friend_select > self.direct_friend_max:
            raise ValueError("candidate_direct_friend_select exceeds friend pool")
        if self.direct_v1_only and not self.direct_two_stage_retrieval:
            # Original v1 aggregated every friend in its fixed friend pool.
            # friend_select only belongs to the optional memory-saving coarse
            # retrieval used when deliberately expanding that pool.
            self.direct_friend_select = self.direct_friend_max
        if self.direct_friend_screen_topl > self.direct_friend_history_max:
            raise ValueError("friend screening Top-L exceeds history maximum")
        if self.direct_freeze_epochs < 0 or not 0 < self.direct_retrieval_lr_scale <= 1:
            raise ValueError("invalid direct-history freeze or learning-rate scale")
        if self.direct_user_topl > self.direct_user_history_max:
            raise ValueError("candidate_direct_user_topl exceeds history maximum")
        if self.direct_friend_topl > self.direct_friend_history_max:
            raise ValueError("candidate_direct_friend_topl exceeds friend-history maximum")
        if min(
            self.direct_screen_consistency_weight,
            self.direct_screen_prior_weight,
        ) < 0:
            raise ValueError("direct-history screening weights must be nonnegative")
        if self.direct_tau <= 0 or self.direct_consistency_weight < 0:
            raise ValueError("invalid direct-history temperature or consistency weight")
        if not 0 < self.direct_init_scale < self.direct_scale_max:
            raise ValueError("direct-history initial scale must lie between 0 and max")
        if self.direct_reg_weight < 0 or self.direct_confidence_weight < 0:
            raise ValueError("direct-history regularization weights must be nonnegative")
        if min(
            self.direct_target_margin_ratio,
            self.direct_max_margin_ratio,
            self.direct_rank_beta,
        ) <= 0:
            raise ValueError("direct-history rank-calibration values must be positive")
        if self.direct_target_margin_ratio > self.direct_max_margin_ratio:
            raise ValueError("direct-history target margin exceeds its maximum")
        if self.direct_correction_reg_weight < 0:
            raise ValueError("direct-history correction regularization must be nonnegative")
        if self.direct_score_scale_floor <= 0:
            raise ValueError("direct-history score-scale floor must be positive")
        if (
            (not self.direct_v1_only and self.direct_router_interests < 2)
            or self.direct_router_tau <= 0
        ):
            raise ValueError("direct-history prototype router configuration is invalid")
        if min(
            self.direct_router_weight,
            self.direct_router_balance_weight,
            self.direct_router_entropy_weight,
        ) < 0:
            raise ValueError("direct-history prototype router weights must be nonnegative")
        if min(
            self.direct_calibration_warmup,
            self.direct_calibration_interval,
            self.direct_calibration_items,
            self.direct_background_items,
        ) <= 0:
            raise ValueError("direct-history calibration sizes must be positive")
        if self.direct_trust_tau <= 0 or self.direct_trust_shrinkage < 0:
            raise ValueError("invalid direct-history trust calibration")
        if self.direct_lift_margin < 0 or self.direct_prior_weight < 0:
            raise ValueError("direct-history lift/prior weights must be nonnegative")
        if self.direct_calibrated_trust and self.direct_hierarchical_prototypes:
            raise ValueError("calibrated trust and hierarchical prototypes are exclusive")
        if self.direct_v1_only and any((
            self.direct_rank_calibrated,
            self.direct_novelty_v1,
            self.direct_calibrated_trust,
            self.direct_hierarchical_prototypes,
        )):
            raise ValueError(
                "pure direct-history v1 cannot combine with later candidate modules"
            )
        if min(
            self.direct_fine_interests,
            self.direct_coarse_interests,
            self.direct_fine_topk,
            self.direct_prototype_warmup,
            self.direct_prototype_interval,
            self.direct_kmeans_iters,
        ) <= 0:
            raise ValueError("hierarchical prototype sizes must be positive")
        if self.direct_fine_interests % self.direct_coarse_interests != 0:
            raise ValueError("fine interests must be divisible by coarse interests")
        if self.direct_fine_topk > self.direct_fine_interests:
            raise ValueError("fine Top-K exceeds the number of fine interests")

        def spread_sample(values, limit):
            values = list(dict.fromkeys(int(x) for x in values))
            if len(values) <= limit:
                return values
            size = len(values)
            return [values[min(size - 1, int((p + 0.5) * size / limit))]
                    for p in range(limit)]

        own_items = torch.zeros(
            (self.user_num, self.direct_user_history_max), dtype=torch.long
        )
        own_mask = torch.zeros_like(own_items, dtype=torch.bool)
        calibration_items = torch.zeros(
            (self.user_num, self.direct_calibration_items), dtype=torch.long
        )
        calibration_mask = torch.zeros_like(calibration_items, dtype=torch.bool)
        friend_history_items = torch.zeros(
            (self.user_num, self.direct_friend_history_max), dtype=torch.long
        )
        friend_history_mask = torch.zeros_like(
            friend_history_items, dtype=torch.bool
        )
        train_items = getattr(data_obj, "train_items", {})
        if not train_items and hasattr(data_obj, "train_data"):
            train_items = {}
            for user_id, item_id in data_obj.train_data:
                train_items.setdefault(int(user_id), []).append(int(item_id))
        self.direct_full_train_items = tuple(
            tuple(dict.fromkeys(int(x) for x in train_items.get(user_id, ())))
            for user_id in range(self.user_num)
        )
        seen_codes = sorted(
            user_id * self.item_num + item_id
            for user_id, history in enumerate(self.direct_full_train_items)
            for item_id in history
        )
        for user_id in range(self.user_num):
            history = train_items.get(user_id, ())
            own = spread_sample(history, self.direct_user_history_max)
            social = spread_sample(history, self.direct_friend_history_max)
            calibration = spread_sample(history, self.direct_calibration_items)
            if own:
                own_items[user_id, :len(own)] = torch.tensor(own, dtype=torch.long)
                own_mask[user_id, :len(own)] = True
            if calibration:
                calibration_items[user_id, :len(calibration)] = torch.tensor(
                    calibration, dtype=torch.long
                )
                calibration_mask[user_id, :len(calibration)] = True
            if social:
                friend_history_items[user_id, :len(social)] = torch.tensor(
                    social, dtype=torch.long
                )
                friend_history_mask[user_id, :len(social)] = True

        adj = self.adj_uu.coalesce()
        adj_indices = adj.indices().detach().cpu()
        adj_values = adj.values().detach().cpu()
        neighbor_lists = [[] for _ in range(self.user_num)]
        for edge_id in range(adj_values.numel()):
            source = int(adj_indices[0, edge_id])
            target = int(adj_indices[1, edge_id])
            if source != target:
                neighbor_lists[source].append(
                    (target, float(abs(adj_values[edge_id])))
                )
        friend_ids = torch.zeros(
            (self.user_num, self.direct_friend_max), dtype=torch.long
        )
        friend_mask = torch.zeros_like(friend_ids, dtype=torch.bool)
        friend_prior = torch.zeros(
            (self.user_num, self.direct_friend_max), dtype=torch.float32
        )
        for user_id, neighbors in enumerate(neighbor_lists):
            if self.direct_v1_only:
                # Preserve v1's stable adjacency order for equal-weight
                # friends.  Coalesced sparse rows are target-id ordered.
                neighbors.sort(key=lambda pair: -pair[1])
            else:
                # Later variants use hash-based tie breaking to avoid a
                # systematic low-user-id bias when normalized edges are equal.
                neighbors.sort(
                    key=lambda pair: (
                        -pair[1],
                        ((pair[0] * 1103515245 + user_id * 12345) & 0x7FFFFFFF),
                    )
                )
            chosen = neighbors[:self.direct_friend_max]
            if not chosen:
                continue
            weights = torch.tensor([x[1] for x in chosen], dtype=torch.float32)
            weights = weights / weights.sum().clamp_min(1e-12)
            count = len(chosen)
            friend_ids[user_id, :count] = torch.tensor(
                [x[0] for x in chosen], dtype=torch.long
            )
            friend_mask[user_id, :count] = True
            friend_prior[user_id, :count] = weights

        self.register_buffer(
            "direct_own_history_items", own_items.to(device), persistent=False
        )
        self.register_buffer(
            "direct_own_history_mask", own_mask.to(device), persistent=False
        )
        self.register_buffer(
            "direct_calibration_item_ids",
            calibration_items.to(device),
            persistent=False,
        )
        self.register_buffer(
            "direct_calibration_mask",
            calibration_mask.to(device),
            persistent=False,
        )
        self.register_buffer(
            "direct_friend_history_items",
            friend_history_items.to(device),
            persistent=False,
        )
        self.register_buffer(
            "direct_friend_history_mask",
            friend_history_mask.to(device),
            persistent=False,
        )
        self.register_buffer(
            "direct_friend_ids", friend_ids.to(device), persistent=False
        )
        self.register_buffer(
            "direct_friend_mask", friend_mask.to(device), persistent=False
        )
        self.register_buffer(
            "direct_friend_prior", friend_prior.to(device), persistent=False
        )
        self.register_buffer(
            "direct_seen_codes",
            torch.tensor(seen_codes, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "direct_friend_trust",
            torch.full(
                (self.user_num, self.direct_friend_max),
                0.5,
                dtype=torch.float32,
                device=device,
            ),
            persistent=not self.direct_v1_only,
        )
        self.register_buffer(
            "direct_friend_background",
            torch.zeros(
                (self.user_num, self.direct_router_interests),
                dtype=torch.float32,
                device=device,
            ),
            persistent=not self.direct_v1_only,
        )
        self.register_buffer(
            "direct_calibration_ready",
            torch.tensor(not self.direct_calibrated_trust, dtype=torch.bool, device=device),
            persistent=not self.direct_v1_only,
        )
        self.register_buffer(
            "direct_calibration_epoch",
            torch.tensor(0, dtype=torch.long, device=device),
            persistent=not self.direct_v1_only,
        )
        self.register_buffer(
            "direct_item_cluster",
            torch.arange(self.item_num, dtype=torch.long, device=device)
            % self.direct_fine_interests,
            persistent=not self.direct_v1_only,
        )
        self.register_buffer(
            "direct_prototype_ready",
            torch.tensor(False, dtype=torch.bool, device=device),
            persistent=not self.direct_v1_only,
        )
        self.register_buffer(
            "direct_prototype_epoch",
            torch.tensor(0, dtype=torch.long, device=device),
            persistent=not self.direct_v1_only,
        )

        d = self.emb_size
        if not self.direct_v1_only:
            self.direct_router_head = nn.Linear(d, self.direct_router_interests)
            nn.init.xavier_uniform_(self.direct_router_head.weight)
            nn.init.zeros_(self.direct_router_head.bias)
        self.direct_query_proj = nn.Linear(d, d, bias=False)
        self.direct_key_proj = nn.Linear(d, d, bias=False)
        self.direct_value_proj = nn.Linear(d, d, bias=False)
        self.direct_friend_proj = nn.Linear(d, d, bias=False)
        for layer in (
            self.direct_query_proj,
            self.direct_key_proj,
            self.direct_value_proj,
            self.direct_friend_proj,
        ):
            nn.init.eye_(layer.weight)
        direct_feature_count = 6 if self.direct_v1_only else 7
        self.direct_evidence_mlp = nn.Sequential(
            nn.Linear(direct_feature_count, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.direct_evidence_mlp[-1].weight)
        nn.init.zeros_(self.direct_evidence_mlp[-1].bias)
        self.direct_gate_mlp = nn.Sequential(
            nn.Linear(direct_feature_count, 32), nn.ReLU(), nn.Linear(32, d)
        )
        nn.init.zeros_(self.direct_gate_mlp[-1].weight)
        nn.init.constant_(self.direct_gate_mlp[-1].bias, -1.0)
        scale_ratio = self.direct_init_scale / self.direct_scale_max
        self.direct_query_proj.weight.register_hook(
            lambda grad: grad * self.direct_retrieval_lr_scale
        )
        self.direct_key_proj.weight.register_hook(
            lambda grad: grad * self.direct_retrieval_lr_scale
        )
        self.direct_scale_logit = nn.Parameter(
            torch.tensor(
                math.log(scale_ratio / (1.0 - scale_ratio)), device=device
            )
        )


    def _direct_router_probability(self, value: torch.Tensor) -> torch.Tensor:
        """Assign embeddings to normalized interest-prototype directions."""
        if self.direct_v1_only:
            return torch.ones(
                *value.shape[:-1], 1, dtype=value.dtype, device=value.device
            )
        value_unit = F.normalize(value, dim=-1)
        prototype_unit = F.normalize(self.direct_router_head.weight, dim=-1)
        router_logits = F.linear(
            value_unit, prototype_unit, self.direct_router_head.bias
        )
        return F.softmax(router_logits / self.direct_router_tau, dim=-1)
    def set_training_epoch(self, epoch: int):
        """Freeze retrieval geometry briefly, then fine-tune it conservatively."""
        super().set_training_epoch(epoch)
        if not self.candidate_direct_history_expert:
            return
        self.direct_current_epoch = int(epoch)
        freeze_until = self.direct_freeze_epochs
        if self.direct_calibrated_trust:
            freeze_until = max(freeze_until, self.direct_calibration_warmup)
        if self.direct_hierarchical_prototypes:
            freeze_until = max(freeze_until, self.direct_prototype_warmup)
        retrieval_trainable = int(epoch) > freeze_until
        self.direct_query_proj.weight.requires_grad_(retrieval_trainable)
        self.direct_key_proj.weight.requires_grad_(retrieval_trainable)

    @torch.no_grad()
    def refresh_direct_calibration(
        self,
        epoch: int,
        force: bool = False,
        user_chunk: int = 256,
    ) -> Optional[Dict[str, float]]:
        """Refresh train-only friend trust and route-conditioned background scores.

        A friend's candidate match is calibrated against its normal support for
        items in the same learned route. Trust is the smoothed mean lift that
        the friend supplies for the user's observed training items. Validation
        and test interactions are never read here.
        """
        if not self.candidate_direct_history_expert or not self.direct_calibrated_trust:
            return None
        epoch = int(epoch)
        if epoch <= self.direct_calibration_warmup:
            return None
        last_epoch = int(self.direct_calibration_epoch.item())
        if (
            not force
            and bool(self.direct_calibration_ready.item())
            and epoch - last_epoch < self.direct_calibration_interval
        ):
            return None

        was_training = self.training
        self.eval()
        _, final_item, _ = self._backbone(
            current_ssl_mode=None, return_aux=True
        )
        item_unit = F.normalize(final_item.detach(), dim=-1)
        route_probability = self._direct_router_probability(final_item.detach())
        route_id = route_probability.argmax(dim=-1)
        history_ids = self.direct_friend_history_items
        history_mask = self.direct_friend_history_mask
        top_count = min(self.direct_friend_topl, history_ids.size(1))

        # b[v,k]: friend v's normal support for arbitrary items in route k.
        background = torch.zeros_like(self.direct_friend_background)
        for route in range(self.direct_router_interests):
            route_items = torch.nonzero(route_id == route, as_tuple=False).flatten()
            if route_items.numel() == 0:
                continue
            sample_count = min(self.direct_background_items, route_items.numel())
            if sample_count < route_items.numel():
                positions = torch.linspace(
                    0,
                    route_items.numel() - 1,
                    steps=sample_count,
                    device=route_items.device,
                ).round().long()
                route_items = route_items[positions]
            route_e = item_unit[route_items]
            for start in range(0, self.user_num, user_chunk):
                stop = min(self.user_num, start + user_chunk)
                ids = history_ids[start:stop]
                valid = history_mask[start:stop]
                history_e = item_unit[ids]
                similarity = torch.einsum("sd,bld->bsl", route_e, history_e)
                expanded_valid = valid.unsqueeze(1).expand_as(similarity)
                similarity = similarity.masked_fill(~expanded_valid, -1e4)
                top_value, top_index = torch.topk(
                    similarity, k=top_count, dim=-1
                )
                top_valid = torch.gather(expanded_valid, -1, top_index)
                match = (
                    top_value.masked_fill(~top_valid, 0.0).sum(dim=-1)
                    / top_valid.sum(dim=-1).clamp_min(1)
                )
                match = match * top_valid.any(dim=-1).to(match.dtype)
                background[start:stop, route] = match.mean(dim=1)

        # A[u,v]: friend v's excess support for user u's observed train items.
        margin = torch.zeros_like(self.direct_friend_trust)
        calibration_count = self.direct_calibration_mask.sum(dim=-1).to(
            item_unit.dtype
        )
        for start in range(0, self.user_num, user_chunk):
            stop = min(self.user_num, start + user_chunk)
            calibration_ids = self.direct_calibration_item_ids[start:stop]
            calibration_valid = self.direct_calibration_mask[start:stop]
            calibration_e = item_unit[calibration_ids]
            calibration_route = route_probability[calibration_ids]
            friend_ids = self.direct_friend_ids[start:stop]
            friend_valid = self.direct_friend_mask[start:stop]
            friend_history_ids = history_ids[friend_ids]
            friend_history_valid = (
                history_mask[friend_ids] & friend_valid.unsqueeze(-1)
            )
            friend_history_e = item_unit[friend_history_ids]
            similarity = torch.einsum(
                "bcd,bfld->bcfl", calibration_e, friend_history_e
            )
            expanded_valid = friend_history_valid.unsqueeze(1).expand_as(similarity)
            similarity = similarity.masked_fill(~expanded_valid, -1e4)
            top_value, top_index = torch.topk(
                similarity, k=top_count, dim=-1
            )
            top_valid = torch.gather(expanded_valid, -1, top_index)
            friend_match = (
                top_value.masked_fill(~top_valid, 0.0).sum(dim=-1)
                / top_valid.sum(dim=-1).clamp_min(1)
            )
            friend_background = background[friend_ids]
            expected_background = torch.einsum(
                "bck,bfk->bcf", calibration_route, friend_background
            )
            pair_valid = (
                calibration_valid.unsqueeze(-1)
                & friend_valid.unsqueeze(1)
                & top_valid.any(dim=-1)
            )
            lift = (friend_match - expected_background) * pair_valid.to(
                friend_match.dtype
            )
            margin[start:stop] = lift.sum(dim=1) / pair_valid.sum(
                dim=1
            ).clamp_min(1)

        edge_valid = self.direct_friend_mask
        if edge_valid.any():
            global_margin = margin[edge_valid].mean()
        else:
            global_margin = torch.zeros((), device=margin.device)
        count = calibration_count.unsqueeze(-1).expand_as(margin)
        smoothed_margin = (
            count * margin + self.direct_trust_shrinkage * global_margin
        ) / (count + self.direct_trust_shrinkage).clamp_min(1e-8)
        trust = torch.sigmoid(smoothed_margin / self.direct_trust_tau)
        trust = trust * edge_valid.to(trust.dtype)

        self.direct_friend_background.copy_(background)
        self.direct_friend_trust.copy_(trust)
        self.direct_calibration_ready.fill_(True)
        self.direct_calibration_epoch.fill_(epoch)
        if was_training:
            self.train()

        valid_trust = trust[edge_valid]
        valid_margin = smoothed_margin[edge_valid]
        route_usage = torch.bincount(
            route_id, minlength=self.direct_router_interests
        ).to(item_unit.dtype) / float(self.item_num)
        return {
            "epoch": float(epoch),
            "trust_mean": float(valid_trust.mean().item()) if valid_trust.numel() else 0.0,
            "trust_min": float(valid_trust.min().item()) if valid_trust.numel() else 0.0,
            "trust_max": float(valid_trust.max().item()) if valid_trust.numel() else 0.0,
            "margin_mean": float(valid_margin.mean().item()) if valid_margin.numel() else 0.0,
            "background_mean": float(background.mean().item()),
            "route_usage_min": float(route_usage.min().item()),
            "route_usage_max": float(route_usage.max().item()),
        }

    @torch.no_grad()
    def refresh_direct_prototypes(
        self,
        epoch: int,
        force: bool = False,
    ) -> Optional[Dict[str, float]]:
        """Cluster item embeddings into paired fine interests using train state only."""
        if (
            not self.candidate_direct_history_expert
            or not self.direct_hierarchical_prototypes
        ):
            return None
        epoch = int(epoch)
        if epoch <= self.direct_prototype_warmup:
            return None
        last_epoch = int(self.direct_prototype_epoch.item())
        if (
            not force
            and bool(self.direct_prototype_ready.item())
            and epoch - last_epoch < self.direct_prototype_interval
        ):
            return None

        was_training = self.training
        self.eval()
        _, final_item, _ = self._backbone(
            current_ssl_mode=None, return_aux=True
        )
        item_unit = F.normalize(final_item.detach(), dim=-1)
        fine_count = self.direct_fine_interests

        # Deterministic farthest-point initialization followed by cosine K-means.
        first = (epoch * 9973) % self.item_num
        centroid_list = [item_unit[first]]
        best_similarity = torch.matmul(item_unit, centroid_list[0])
        for _ in range(1, fine_count):
            next_index = torch.argmin(best_similarity)
            next_centroid = item_unit[next_index]
            centroid_list.append(next_centroid)
            best_similarity = torch.maximum(
                best_similarity, torch.matmul(item_unit, next_centroid)
            )
        centroids = torch.stack(centroid_list, dim=0)
        assignment = torch.zeros(
            self.item_num, dtype=torch.long, device=item_unit.device
        )
        for _ in range(self.direct_kmeans_iters):
            similarity = torch.matmul(item_unit, centroids.t())
            assignment = similarity.argmax(dim=-1)
            cluster_sum = torch.zeros_like(centroids)
            cluster_sum.index_add_(0, assignment, item_unit)
            cluster_count = torch.bincount(
                assignment, minlength=fine_count
            ).to(item_unit.dtype)
            updated = cluster_sum / cluster_count.clamp_min(1.0).unsqueeze(-1)
            updated = F.normalize(updated, dim=-1)
            centroids = torch.where(
                (cluster_count > 0).unsqueeze(-1), updated, centroids
            )

        # Pair the nearest fine centroids so four coarse prototypes retain two
        # distinct modes instead of averaging an arbitrary pair.
        centroid_similarity = torch.matmul(centroids, centroids.t())
        remap = torch.empty(fine_count, dtype=torch.long, device=item_unit.device)
        unused = list(range(fine_count))
        group_size = fine_count // self.direct_coarse_interests
        for coarse in range(self.direct_coarse_interests):
            anchor = unused.pop(0)
            group = [anchor]
            while len(group) < group_size and unused:
                group_tensor = torch.tensor(
                    unused, dtype=torch.long, device=item_unit.device
                )
                group_score = centroid_similarity[anchor, group_tensor]
                chosen_position = int(torch.argmax(group_score).item())
                group.append(unused.pop(chosen_position))
            for offset, old_cluster in enumerate(group):
                remap[old_cluster] = coarse * group_size + offset
        assignment = remap[assignment]

        self.direct_item_cluster.copy_(assignment)
        self.direct_prototype_ready.fill_(True)
        self.direct_prototype_epoch.fill_(epoch)
        if was_training:
            self.train()

        usage = torch.bincount(
            assignment, minlength=fine_count
        ).to(item_unit.dtype) / float(self.item_num)
        return {
            "epoch": float(epoch),
            "fine_usage_min": float(usage.min().item()),
            "fine_usage_max": float(usage.max().item()),
            "fine_usage_entropy": float(
                -(usage * usage.clamp_min(1e-8).log()).sum().item()
            ),
        }

    def _backbone(
        self,
        current_ssl_mode: Optional[str] = None,
        return_aux: bool = False,
    ):
        if not (
            self.candidate_retrieval_expert
            or self.candidate_direct_history_expert
        ):
            return super()._backbone(current_ssl_mode, return_aux)

        # The preserved backbone sees candidate_subspace=False so it computes
        # exactly the original ODSR representation and auxiliary decomposition.
        self.candidate_subspace = False
        try:
            output = super()._backbone(current_ssl_mode, return_aux)
        finally:
            self.candidate_subspace = True
        if not return_aux:
            return output

        final_user, final_item, aux = output
        if self.candidate_direct_history_expert:
            aux["retrieval_user_anchor"] = final_user
            if (
                self.direct_hierarchical_prototypes
                and bool(self.direct_prototype_ready.item())
            ):
                fine_count = self.direct_fine_interests
                coarse_count = self.direct_coarse_interests
                group_size = fine_count // coarse_count
                assignment = F.one_hot(
                    self.direct_item_cluster, num_classes=fine_count
                ).to(final_item.dtype)
                weighted_item = (
                    final_item.unsqueeze(1) * assignment.unsqueeze(-1)
                ).reshape(self.item_num, fine_count * self.emb_size)
                fine_sum = torch.sparse.mm(
                    self.adj_ui, weighted_item
                ).reshape(self.user_num, fine_count, self.emb_size)
                fine_mass = torch.sparse.mm(self.adj_ui, assignment)
                fine_interest = fine_sum / fine_mass.clamp_min(1e-8).unsqueeze(-1)
                grouped_sum = fine_sum.reshape(
                    self.user_num,
                    coarse_count,
                    group_size,
                    self.emb_size,
                ).sum(dim=2)
                coarse_mass = fine_mass.reshape(
                    self.user_num, coarse_count, group_size
                ).sum(dim=2)
                coarse_interest = grouped_sum / coarse_mass.clamp_min(
                    1e-8
                ).unsqueeze(-1)
                aux.update(
                    {
                        "direct_fine_sum": fine_sum,
                        "direct_fine_interest": fine_interest,
                        "direct_fine_mass": fine_mass,
                        "direct_coarse_interest": coarse_interest,
                        "direct_coarse_mass": coarse_mass,
                    }
                )
            return final_user, final_item, aux
        k = self.retrieval_num_interests
        d = self.emb_size
        head_probability = F.softmax(
            self.retrieval_item_head(final_item) / self.retrieval_assignment_tau,
            dim=-1,
        )

        weighted_item = (
            final_item.unsqueeze(1) * head_probability.unsqueeze(-1)
        ).reshape(self.item_num, k * d)
        own_sum = torch.sparse.mm(self.adj_ui, weighted_item).reshape(
            self.user_num, k, d
        )
        own_mass = torch.sparse.mm(self.adj_ui, head_probability)
        own_interest = own_sum / own_mass.clamp_min(1e-8).unsqueeze(-1)

        # Aggregate supported neighbor interests, rather than treating empty
        # heads as real zero-valued evidence.
        social_numerator = torch.sparse.mm(
            self.adj_uu_neighbors,
            (own_interest * own_mass.unsqueeze(-1)).reshape(self.user_num, k * d),
        ).reshape(self.user_num, k, d)
        social_mass = torch.sparse.mm(self.adj_uu_neighbors, own_mass)
        social_interest = social_numerator / social_mass.clamp_min(1e-8).unsqueeze(-1)

        retrieval_aux = {
            "retrieval_user_anchor": final_user,
            "retrieval_own_interest": own_interest,
            "retrieval_own_mass": own_mass,
            "retrieval_social_interest": social_interest,
            "retrieval_social_mass": social_mass,
            "retrieval_head_probability": head_probability,
            "retrieval_head_usage": head_probability.mean(dim=0),
        }
        if self.retrieval_specificity:
            # A prototype mean alone cannot distinguish a compact interest from
            # a broad average. The normalized weighted variance is a scalar
            # radius in [0, 1] and needs no per-candidate history scan.
            item_square = final_item.square().sum(dim=-1, keepdim=True)
            own_second_sum = torch.sparse.mm(
                self.adj_ui, head_probability * item_square
            )
            own_second = own_second_sum / own_mass.clamp_min(1e-8)
            own_variance = (
                own_second - own_interest.square().sum(dim=-1)
            ).clamp_min(0.0)
            own_dispersion = (
                own_variance / own_second.clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            social_second_sum = torch.sparse.mm(
                self.adj_uu_neighbors, own_second_sum
            )
            social_second = social_second_sum / social_mass.clamp_min(1e-8)
            social_variance = (
                social_second - social_interest.square().sum(dim=-1)
            ).clamp_min(0.0)
            social_dispersion = (
                social_variance / social_second.clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            retrieval_aux.update(
                {
                    "retrieval_own_dispersion": own_dispersion,
                    "retrieval_social_dispersion": social_dispersion,
                }
            )
        aux.update(retrieval_aux)
        return final_user, final_item, aux

    @staticmethod
    def _candidate_user_view(value: torch.Tensor, users: torch.Tensor, item_e: torch.Tensor):
        selected = value[users]
        if item_e.dim() == 3:
            selected = selected.unsqueeze(1)
        return selected

    def score_matrix(
        self,
        users: torch.Tensor,
        final_user: Optional[torch.Tensor] = None,
        final_item: Optional[torch.Tensor] = None,
        aux: Optional[Dict[str, torch.Tensor]] = None,
        item_chunk: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.candidate_direct_history_expert:
            return super().score_matrix(
                users, final_user, final_item, aux, item_chunk
            )
        if final_user is None or final_item is None or aux is None:
            final_user, final_item, aux = self._backbone(
                current_ssl_mode=None, return_aux=True
            )
        base_scores = torch.matmul(final_user[users], final_item.t())
        selection_scores = base_scores.detach().clone()
        # Select the shortlist after excluding training interactions.  The
        # evaluation utility masks them again on the returned full score matrix.
        for row, user_id in enumerate(users.detach().cpu().tolist()):
            seen = self.direct_full_train_items[int(user_id)]
            if seen:
                selection_scores[row, list(seen)] = -torch.inf
        top_count = min(self.direct_eval_topm, self.item_num)
        shortlist = torch.topk(selection_scores, k=top_count, dim=1).indices
        correction, _ = self._direct_history_correction(
            users, shortlist, final_item, aux, detach_backbone=False
        )
        scores = base_scores.clone()
        scores.scatter_add_(1, shortlist, correction)
        return scores

    def _hierarchical_prototype_correction(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        final_item: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        detach_backbone: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Use all train histories through coarse-to-fine interest prototypes.

        Coarse prototypes screen the friend pool. Fine prototypes construct the
        candidate-specific user and friend views. The first unselected friend is
        the post-selection background, and ``-log(S)`` prevents S friends from
        receiving S times the prior odds of the single null expert.
        """
        item_source = final_item.detach() if detach_backbone else final_item
        item_e = item_source[items]
        squeeze_candidate = item_e.dim() == 2
        if squeeze_candidate:
            item_e = item_e.unsqueeze(1)
            item_ids = items.unsqueeze(1)
        else:
            item_ids = items
        batch_size, candidate_count, dim = item_e.shape
        fine_count = self.direct_fine_interests
        coarse_count = self.direct_coarse_interests
        group_size = fine_count // coarse_count
        fine_topk = min(self.direct_fine_topk, fine_count)

        item_unit = F.normalize(item_e, dim=-1)
        query_unit = F.normalize(self.direct_query_proj(item_e), dim=-1)
        candidate_fine_id = self.direct_item_cluster[item_ids]
        candidate_coarse_id = torch.div(
            candidate_fine_id, group_size, rounding_mode="floor"
        )
        item_router_probability = F.one_hot(
            candidate_coarse_id, num_classes=coarse_count
        ).to(item_e.dtype)

        user_anchor = aux["retrieval_user_anchor"][users]
        fine_sum = aux["direct_fine_sum"][users]
        fine_mass = aux["direct_fine_mass"][users]
        if detach_backbone:
            user_anchor = user_anchor.detach()
            fine_sum = fine_sum.detach()
            fine_mass = fine_mass.detach()
        user_anchor = user_anchor.unsqueeze(1).expand(-1, candidate_count, -1)
        fine_sum = fine_sum.unsqueeze(1).expand(-1, candidate_count, -1, -1)
        fine_mass = fine_mass.unsqueeze(1).expand(-1, candidate_count, -1)

        # Leave the target item out of its owner's prototype. This prevents a
        # training positive from retrieving itself while preserving all other
        # observed interactions in the prototype memory.
        candidate_code = users.unsqueeze(1) * self.item_num + item_ids
        if self.direct_seen_codes.numel() > 0:
            position = torch.searchsorted(self.direct_seen_codes, candidate_code)
            safe_position = position.clamp_max(self.direct_seen_codes.numel() - 1)
            candidate_seen = (
                position < self.direct_seen_codes.numel()
            ) & (self.direct_seen_codes[safe_position] == candidate_code)
        else:
            candidate_seen = torch.zeros_like(candidate_code, dtype=torch.bool)
        candidate_assignment = F.one_hot(
            candidate_fine_id, num_classes=fine_count
        ).to(item_e.dtype)
        leave_weight = candidate_assignment * candidate_seen.unsqueeze(-1).to(
            item_e.dtype
        )
        own_sum = fine_sum - leave_weight.unsqueeze(-1) * item_e.unsqueeze(-2)
        own_mass = (fine_mass - leave_weight).clamp_min(0.0)
        own_interest = own_sum / own_mass.clamp_min(1e-8).unsqueeze(-1)
        own_valid = own_mass > 0
        own_key = F.normalize(self.direct_key_proj(own_interest), dim=-1)
        own_similarity = torch.sum(query_unit.unsqueeze(-2) * own_key, dim=-1)
        own_similarity = own_similarity.masked_fill(~own_valid, -1e4)
        own_top_value, own_top_index = torch.topk(
            own_similarity, k=fine_topk, dim=-1
        )
        own_top_valid = torch.gather(own_valid, -1, own_top_index)
        own_weight = F.softmax(own_top_value / self.direct_tau, dim=-1)
        own_weight = own_weight * own_top_valid.to(own_weight.dtype)
        own_weight = own_weight / own_weight.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        own_value = self.direct_value_proj(own_interest)
        own_selected = torch.gather(
            own_value,
            2,
            own_top_index.unsqueeze(-1).expand(-1, -1, -1, dim),
        )
        local_interest = torch.sum(own_weight.unsqueeze(-1) * own_selected, dim=-2)
        own_available = own_top_valid.any(dim=-1)
        local_interest = torch.where(
            own_available.unsqueeze(-1), local_interest, user_anchor
        )
        local_axis = F.normalize(local_interest, dim=-1)
        own_relevance = torch.sum(
            own_weight * own_top_value.masked_fill(~own_top_valid, 0.0), dim=-1
        )
        own_selected_mass = torch.gather(own_mass, -1, own_top_index)
        own_support = torch.sum(
            own_weight * (own_selected_mass / (own_selected_mass + 1.0)), dim=-1
        )
        user_item_cosine = torch.sum(local_axis * item_unit, dim=-1)

        friend_pool_ids = self.direct_friend_ids[users]
        friend_pool_mask = self.direct_friend_mask[users]
        friend_pool_prior = self.direct_friend_prior[users]
        coarse_interest = aux["direct_coarse_interest"]
        coarse_mass = aux["direct_coarse_mass"]
        if detach_backbone:
            coarse_interest = coarse_interest.detach()
            coarse_mass = coarse_mass.detach()
        pool_coarse_interest = coarse_interest[friend_pool_ids]
        pool_coarse_mass = coarse_mass[friend_pool_ids]

        # Coarse screening has fixed O(B*N*F*C*D) cost regardless of history
        # length. The fifth-ranked friend becomes the post-selection baseline.
        with torch.no_grad():
            pool_coarse_key = F.normalize(
                self.direct_key_proj(pool_coarse_interest), dim=-1
            )
            screen_similarity = torch.einsum(
                "bnd,bfcd->bnfc", query_unit.detach(), pool_coarse_key
            )
            screen_valid = (
                friend_pool_mask.unsqueeze(-1) & (pool_coarse_mass > 0)
            ).unsqueeze(1).expand_as(screen_similarity)
            screen_similarity = screen_similarity.masked_fill(~screen_valid, -1e4)
            screen_match, screen_coarse_index = screen_similarity.max(dim=-1)
            screen_eligible = screen_valid.any(dim=-1)
            selected_screen_mass = torch.gather(
                pool_coarse_mass.unsqueeze(1).expand(
                    -1, candidate_count, -1, -1
                ),
                3,
                screen_coarse_index.unsqueeze(-1),
            ).squeeze(-1)
            screen_support = selected_screen_mass / (selected_screen_mass + 1.0)
            relative_prior = friend_pool_prior / friend_pool_prior.amax(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            screen_score = screen_match + self.direct_prior_weight * torch.log(
                relative_prior.unsqueeze(1).clamp_min(1e-8)
            )
            screen_score = screen_score.masked_fill(~screen_eligible, -1e4)
            selected_count = min(self.direct_friend_select, screen_score.size(-1))
            ranked_count = min(selected_count + 1, screen_score.size(-1))
            ranked_index = torch.topk(
                screen_score, k=ranked_count, dim=-1
            ).indices
            selected_friend_index = ranked_index[..., :selected_count]
            if ranked_count > selected_count:
                background_index = ranked_index[..., selected_count:selected_count + 1]
                background_match = torch.gather(
                    screen_match, 2, background_index
                ).squeeze(-1)
                background_valid = torch.gather(
                    screen_eligible, 2, background_index
                ).squeeze(-1)
                selection_background = torch.where(
                    background_valid,
                    background_match,
                    torch.full_like(background_match, self.direct_match_threshold),
                )
            else:
                selection_background = torch.full_like(
                    screen_match[..., 0], self.direct_match_threshold
                )

        expanded_pool_ids = friend_pool_ids.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        expanded_pool_mask = friend_pool_mask.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        expanded_pool_prior = friend_pool_prior.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        selected_friend_ids = torch.gather(
            expanded_pool_ids, 2, selected_friend_index
        )
        selected_friend_mask = torch.gather(
            expanded_pool_mask, 2, selected_friend_index
        ) & torch.gather(screen_eligible, 2, selected_friend_index)
        selected_friend_prior = torch.gather(
            expanded_pool_prior, 2, selected_friend_index
        )
        selected_screen_match = torch.gather(
            screen_match, 2, selected_friend_index
        )
        selected_screen_support = torch.gather(
            screen_support, 2, selected_friend_index
        )

        all_fine_interest = aux["direct_fine_interest"]
        all_fine_mass = aux["direct_fine_mass"]
        if detach_backbone:
            all_fine_interest = all_fine_interest.detach()
            all_fine_mass = all_fine_mass.detach()
        friend_fine_interest = all_fine_interest[selected_friend_ids]
        friend_fine_mass = all_fine_mass[selected_friend_ids]
        friend_fine_key = F.normalize(
            self.direct_key_proj(friend_fine_interest), dim=-1
        )
        friend_fine_similarity = torch.sum(
            query_unit.unsqueeze(-2).unsqueeze(-2) * friend_fine_key,
            dim=-1,
        )
        friend_fine_valid = (
            (friend_fine_mass > 0) & selected_friend_mask.unsqueeze(-1)
        )
        friend_fine_similarity = friend_fine_similarity.masked_fill(
            ~friend_fine_valid, -1e4
        )
        friend_top_value, friend_top_index = torch.topk(
            friend_fine_similarity, k=fine_topk, dim=-1
        )
        friend_top_valid = torch.gather(
            friend_fine_valid, -1, friend_top_index
        )
        friend_prototype_weight = F.softmax(
            friend_top_value / self.direct_tau, dim=-1
        )
        friend_prototype_weight = (
            friend_prototype_weight
            * friend_top_valid.to(friend_prototype_weight.dtype)
        )
        friend_prototype_weight = friend_prototype_weight / (
            friend_prototype_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        friend_value = self.direct_value_proj(friend_fine_interest)
        friend_selected = torch.gather(
            friend_value,
            3,
            friend_top_index.unsqueeze(-1).expand(-1, -1, -1, -1, dim),
        )
        friend_local = torch.sum(
            friend_prototype_weight.unsqueeze(-1) * friend_selected, dim=-2
        )
        friend_available = friend_top_valid.any(dim=-1)
        friend_match = torch.sum(
            friend_prototype_weight
            * friend_top_value.masked_fill(~friend_top_valid, 0.0),
            dim=-1,
        )
        friend_selected_mass = torch.gather(
            friend_fine_mass, -1, friend_top_index
        )
        multi_support = torch.sum(
            friend_prototype_weight
            * (friend_selected_mass / (friend_selected_mass + 1.0)),
            dim=-1,
        )
        friend_lift = friend_match - selection_background.unsqueeze(-1)
        friend_prior = selected_friend_prior * friend_available.to(item_e.dtype)
        friend_prior = friend_prior / friend_prior.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        friend_unit = F.normalize(friend_local, dim=-1)
        friend_projected = F.normalize(
            self.direct_friend_proj(friend_local), dim=-1
        )
        relevance = torch.sum(
            friend_projected * query_unit.unsqueeze(-2), dim=-1
        )
        consistency = torch.sum(friend_unit * local_axis.unsqueeze(-2), dim=-1)
        global_axis = F.normalize(user_anchor, dim=-1)
        local_global_consistency = torch.sum(local_axis * global_axis, dim=-1)
        friend_global_consistency = torch.sum(
            friend_unit * global_axis.unsqueeze(-2), dim=-1
        )
        parallel_scale = torch.sum(
            friend_local * local_axis.unsqueeze(-2), dim=-1, keepdim=True
        )
        friend_parallel = parallel_scale * local_axis.unsqueeze(-2)
        friend_orthogonal = friend_local - friend_parallel
        gate_features = torch.stack(
            [
                relevance,
                consistency,
                multi_support,
                friend_prior,
                selected_screen_support,
                user_item_cosine.unsqueeze(-1).expand_as(relevance),
                local_global_consistency.unsqueeze(-1).expand_as(relevance),
            ],
            dim=-1,
        )
        friend_gate = torch.sigmoid(self.direct_gate_mlp(gate_features))
        friend_clean = friend_parallel + friend_gate * friend_orthogonal
        clean_item_cosine = torch.sum(
            F.normalize(friend_clean, dim=-1) * item_unit.unsqueeze(-2), dim=-1
        )
        evidence_features = torch.stack(
            [
                relevance,
                consistency,
                clean_item_cosine,
                multi_support,
                friend_prior,
                selected_screen_support,
                friend_global_consistency,
            ],
            dim=-1,
        )
        evidence_residual = self.direct_residual_max * torch.tanh(
            self.direct_evidence_mlp(evidence_features).squeeze(-1)
        )
        evidence = relevance + self.direct_consistency_weight * consistency + evidence_residual

        # Only independently useful evidence competes with the null expert.
        # Dividing total friend odds by S removes the many-friends prior bias.
        friend_logits = (
            (friend_lift - self.direct_lift_margin) / self.direct_tau
            + torch.log(multi_support.clamp_min(1e-8))
            + self.direct_prior_weight * torch.log(friend_prior.clamp_min(1e-8))
            - math.log(float(max(1, selected_count)))
        )
        friend_logits = friend_logits.masked_fill(~friend_available, -1e4)
        null_logit = torch.zeros_like(friend_logits[..., :1])
        choice_probability = F.softmax(
            torch.cat([friend_logits, null_logit], dim=-1), dim=-1
        )
        friend_weight = choice_probability[..., :-1]
        null_probability = choice_probability[..., -1]
        candidate_mass = friend_weight.sum(dim=-1)
        conditional_weight = friend_weight / candidate_mass.unsqueeze(-1).clamp_min(
            1e-8
        )
        candidate_social = torch.sum(
            conditional_weight.unsqueeze(-1) * friend_clean, dim=-2
        )
        candidate_social_cosine = torch.sum(
            F.normalize(candidate_social, dim=-1) * item_unit, dim=-1
        )
        candidate_advantage = candidate_social_cosine - user_item_cosine
        candidate_scale = self.direct_scale_max * torch.sigmoid(
            self.direct_scale_logit
        )
        candidate_score = candidate_mass * candidate_scale * candidate_social_cosine

        weighted_relevance = torch.sum(conditional_weight * relevance, dim=-1)
        weighted_consistency = torch.sum(conditional_weight * consistency, dim=-1)
        weighted_support = torch.sum(conditional_weight * multi_support, dim=-1)
        weighted_route_support = torch.sum(
            conditional_weight * selected_screen_support, dim=-1
        )
        weighted_lift = torch.sum(conditional_weight * friend_lift, dim=-1)
        weighted_parallel = torch.sum(
            conditional_weight.unsqueeze(-1) * friend_parallel, dim=-2
        )
        weighted_orthogonal = torch.sum(
            conditional_weight.unsqueeze(-1) * friend_orthogonal, dim=-2
        )
        friend_trust = friend_available.to(item_e.dtype)

        details = {
            "candidate_score": candidate_score,
            "candidate_scale": candidate_scale.expand_as(candidate_score),
            "candidate_mass": candidate_mass,
            "social_available": friend_available.any(dim=-1).to(item_e.dtype),
            "null_probability": null_probability,
            "friend_weight": friend_weight,
            "local_weight": own_weight,
            "relevance": relevance,
            "consistency": consistency,
            "soft_consistency": torch.sigmoid(consistency / self.direct_tau),
            "support": multi_support,
            "route_support": selected_screen_support,
            "friend_lift": friend_lift,
            "friend_trust": friend_trust,
            "evidence": evidence,
            "evidence_residual": evidence_residual,
            "candidate_parallel": weighted_parallel,
            "candidate_orthogonal": weighted_orthogonal,
            "candidate_gate": friend_gate.mean(dim=(-1, -2)),
            "candidate_clean": candidate_social,
            "candidate_cosine": candidate_social_cosine,
            "candidate_router_probability": item_router_probability,
            "candidate_social_cosine": candidate_social_cosine,
            "candidate_advantage": candidate_advantage,
            "user_item_cosine": user_item_cosine,
            "local_global_consistency": local_global_consistency,
            "weighted_relevance": weighted_relevance,
            "weighted_consistency": weighted_consistency,
            "weighted_support": weighted_support,
            "weighted_route_support": weighted_route_support,
            "weighted_lift": weighted_lift,
            "weighted_trust": torch.sum(
                conditional_weight * friend_trust, dim=-1
            ),
            "own_relevance": own_relevance,
            "own_support": own_support,
            "friend_available_count": friend_available.to(item_e.dtype).sum(dim=-1),
            "friend_pool_available_count": screen_eligible.to(item_e.dtype).sum(dim=-1),
            "friend_pool_support_mean": screen_support.mean(dim=-1),
            "friend_pool_route_mean": screen_match.masked_fill(
                ~screen_eligible, 0.0
            ).sum(dim=-1) / screen_eligible.sum(dim=-1).clamp_min(1),
            "selection_background": selection_background,
            "raw_delta": candidate_score,
        }
        if squeeze_candidate:
            candidate_score = candidate_score[:, 0]
            details = {
                key: (
                    value[:, 0]
                    if value.dim() >= 2 and value.size(1) == 1
                    else value
                )
                for key, value in details.items()
            }
        return candidate_score, details

    def _direct_history_correction(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        final_item: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        detach_backbone: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Retrieve raw user/friend histories and filter friends before aggregation."""
        if (
            self.direct_hierarchical_prototypes
            and bool(self.direct_prototype_ready.item())
        ):
            return self._hierarchical_prototype_correction(
                users, items, final_item, aux, detach_backbone
            )
        item_source = final_item.detach() if detach_backbone else final_item
        item_e = item_source[items]
        squeeze_candidate = item_e.dim() == 2
        if squeeze_candidate:
            item_e = item_e.unsqueeze(1)
            item_ids = items.unsqueeze(1)
        else:
            item_ids = items
        batch_size, candidate_count, dim = item_e.shape
        item_unit = F.normalize(item_e, dim=-1)
        query_unit = F.normalize(self.direct_query_proj(item_e), dim=-1)
        item_router_probability = self._direct_router_probability(item_e)

        own_ids = self.direct_own_history_items[users]
        own_valid_base = self.direct_own_history_mask[users]
        # The training positive belongs to H_u.  Excluding the candidate itself
        # prevents a trivial train-only identity match that cannot occur at test.
        own_valid = (
            own_valid_base.unsqueeze(1)
            & (own_ids.unsqueeze(1) != item_ids.unsqueeze(-1))
        )
        own_top_count = min(self.direct_user_topl, own_ids.size(-1))
        if self.direct_two_stage_retrieval:
            # The enlarged history pool is only used to choose a shortlist.
            # Re-encoding that shortlist below keeps the trainable query, key,
            # value and router paths connected to the final ranking loss.
            with torch.no_grad():
                coarse_own_e = item_source.detach()[own_ids]
                coarse_own_key = F.normalize(
                    self.direct_key_proj(coarse_own_e), dim=-1
                )
                coarse_own_router = self._direct_router_probability(coarse_own_e)
                coarse_own_raw = torch.einsum(
                    "bnd,bld->bnl", query_unit.detach(), coarse_own_key
                )
                coarse_own_route = torch.einsum(
                    "bnk,blk->bnl",
                    item_router_probability.detach(),
                    coarse_own_router,
                )
                coarse_own_score = (
                    coarse_own_raw
                    + self.direct_router_weight
                    * torch.log(
                        self.direct_router_interests * coarse_own_route + 1e-8
                    )
                )
                coarse_own_score = coarse_own_score.masked_fill(
                    ~own_valid, -1e4
                )
                own_top_index = torch.topk(
                    coarse_own_score, k=own_top_count, dim=-1
                ).indices
            own_top_valid = torch.gather(own_valid, -1, own_top_index)
            own_selected_ids = torch.gather(
                own_ids.unsqueeze(1).expand(-1, candidate_count, -1),
                2,
                own_top_index,
            )
            own_selected_e = item_source[own_selected_ids]
            own_selected_key = F.normalize(
                self.direct_key_proj(own_selected_e), dim=-1
            )
            own_selected_router = self._direct_router_probability(
                own_selected_e
            )
            own_top_raw = torch.sum(
                query_unit.unsqueeze(-2) * own_selected_key, dim=-1
            )
            own_top_route = torch.sum(
                item_router_probability.unsqueeze(-2) * own_selected_router,
                dim=-1,
            )
            own_top_value = (
                own_top_raw
                + self.direct_router_weight
                * torch.log(
                    self.direct_router_interests * own_top_route + 1e-8
                )
            )
            own_top_value = own_top_value.masked_fill(~own_top_valid, -1e4)
            own_selected = self.direct_value_proj(own_selected_e)
        else:
            own_e = item_source[own_ids]
            own_key = F.normalize(self.direct_key_proj(own_e), dim=-1)
            own_router_probability = self._direct_router_probability(own_e)
            own_raw_similarity = torch.einsum(
                "bnd,bld->bnl", query_unit, own_key
            )
            own_route_compatibility = torch.einsum(
                "bnk,blk->bnl", item_router_probability, own_router_probability
            )
            own_route_bias = torch.log(
                self.direct_router_interests * own_route_compatibility + 1e-8
            )
            own_similarity = (
                own_raw_similarity + self.direct_router_weight * own_route_bias
            )
            own_similarity = own_similarity.masked_fill(~own_valid, -1e4)
            own_top_value, own_top_index = torch.topk(
                own_similarity, k=own_top_count, dim=-1
            )
            own_top_valid = torch.gather(own_valid, -1, own_top_index)
            own_selected_ids = torch.gather(
                own_ids.unsqueeze(1).expand(-1, candidate_count, -1),
                2,
                own_top_index,
            )
            own_value = self.direct_value_proj(own_e)
            own_selected = torch.gather(
                own_value.unsqueeze(1).expand(-1, candidate_count, -1, -1),
                2,
                own_top_index.unsqueeze(-1).expand(-1, -1, -1, dim),
            )
        own_weight = F.softmax(own_top_value / self.direct_tau, dim=-1)
        own_weight = own_weight * own_top_valid.to(own_weight.dtype)
        own_weight = own_weight / own_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        local_interest = torch.sum(
            own_weight.unsqueeze(-1) * own_selected, dim=-2
        )
        own_available = own_top_valid.any(dim=-1)
        user_anchor = aux["retrieval_user_anchor"][users]
        if detach_backbone:
            user_anchor = user_anchor.detach()
        user_anchor = user_anchor.unsqueeze(1).expand(-1, candidate_count, -1)
        local_interest = torch.where(
            own_available.unsqueeze(-1), local_interest, user_anchor
        )
        local_axis = F.normalize(local_interest, dim=-1)
        own_support = own_top_valid.to(item_e.dtype).mean(dim=-1)
        user_item_cosine = torch.sum(local_axis * item_unit, dim=-1)
        own_relevance = torch.sum(own_weight * own_top_value, dim=-1)

        # Stage 1 screens a broader friend pool with raw cosine scores under
        # no_grad.  Only the candidate-specific shortlist enters the trainable
        # branch, avoiding a [B,N,pool,L,D] autograd tensor.
        friend_pool_ids = self.direct_friend_ids[users]
        friend_pool_mask = self.direct_friend_mask[users]
        friend_pool_prior = self.direct_friend_prior[users]
        friend_pool_trust = self.direct_friend_trust[users]
        friend_pool_background = self.direct_friend_background[friend_pool_ids]
        pool_history_ids = self.direct_friend_history_items[friend_pool_ids]
        pool_history_valid = (
            self.direct_friend_history_mask[friend_pool_ids]
            & friend_pool_mask.unsqueeze(-1)
        )
        with torch.no_grad():
            # Score the large friend pool in small friend chunks.  This avoids
            # materializing [B, friend_pool, history_pool, D] for all friends
            # at once, while retaining the exact global Top-K selection.
            relative_prior = friend_pool_prior / friend_pool_prior.amax(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            screen_retrieve_count = min(
                max(
                    self.direct_friend_screen_topl,
                    (
                        self.direct_friend_topl
                        if self.direct_two_stage_retrieval
                        else self.direct_friend_screen_topl
                    ),
                ),
                pool_history_ids.size(-1),
            )
            screen_top_count = min(
                self.direct_friend_screen_topl, screen_retrieve_count
            )
            screen_chunk = min(
                self.direct_friend_screen_chunk, friend_pool_ids.size(-1)
            )
            score_parts = []
            eligible_parts = []
            support_parts = []
            route_support_parts = []
            match_parts = []
            user_consistency_parts = []
            history_index_parts = []
            for friend_start in range(0, friend_pool_ids.size(-1), screen_chunk):
                friend_stop = min(
                    friend_start + screen_chunk, friend_pool_ids.size(-1)
                )
                chunk_history_ids = pool_history_ids[:, friend_start:friend_stop]
                chunk_history_valid = pool_history_valid[
                    :, friend_start:friend_stop
                ]
                chunk_history_e = item_source.detach()[chunk_history_ids]
                chunk_history_unit = F.normalize(chunk_history_e, dim=-1)
                chunk_router_probability = self._direct_router_probability(
                    chunk_history_e
                )
                chunk_raw_similarity = torch.einsum(
                    "bnd,bfld->bnfl", item_unit.detach(), chunk_history_unit
                )
                chunk_route_compatibility = torch.einsum(
                    "bnk,bflk->bnfl",
                    item_router_probability.detach(),
                    chunk_router_probability,
                )
                chunk_similarity = (
                    chunk_raw_similarity
                    + self.direct_router_weight
                    * torch.log(
                        self.direct_router_interests
                        * chunk_route_compatibility
                        + 1e-8
                    )
                )
                chunk_valid = chunk_history_valid.unsqueeze(1).expand(
                    -1, candidate_count, -1, -1
                )
                chunk_similarity = chunk_similarity.masked_fill(
                    ~chunk_valid, -1e4
                )
                chunk_raw_similarity = chunk_raw_similarity.masked_fill(
                    ~chunk_valid, -1e4
                )
                chunk_top_value_all, chunk_top_index_all = torch.topk(
                    chunk_similarity, k=screen_retrieve_count, dim=-1
                )
                chunk_top_value = chunk_top_value_all[..., :screen_top_count]
                chunk_top_index = chunk_top_index_all[..., :screen_top_count]
                chunk_top_valid = torch.gather(
                    chunk_valid, -1, chunk_top_index
                )
                chunk_top_raw = torch.gather(
                    chunk_raw_similarity, -1, chunk_top_index
                )
                chunk_top_route = torch.gather(
                    chunk_route_compatibility, -1, chunk_top_index
                )
                chunk_support = torch.sigmoid(
                    (chunk_top_raw - self.direct_match_threshold)
                    / self.direct_tau
                ) * chunk_top_valid.to(chunk_top_value.dtype)
                chunk_support = (
                    chunk_support.sum(dim=-1) / float(screen_top_count)
                )
                chunk_route_support = (
                    chunk_top_route
                    * chunk_top_valid.to(chunk_top_route.dtype)
                ).sum(dim=-1) / float(screen_top_count)
                chunk_eligible = chunk_top_valid.any(dim=-1)
                chunk_match = (
                    chunk_top_raw.masked_fill(~chunk_top_valid, 0.0).sum(dim=-1)
                    / chunk_top_valid.sum(dim=-1).clamp_min(1)
                )
                if self.direct_user_consistent_screen:
                    chunk_selected_e = torch.gather(
                        chunk_history_e.unsqueeze(1).expand(
                            -1, candidate_count, -1, -1, -1
                        ),
                        3,
                        chunk_top_index.unsqueeze(-1).expand(
                            -1, -1, -1, -1, dim
                        ),
                    )
                    chunk_selected_value = self.direct_value_proj(
                        chunk_selected_e
                    )
                    chunk_screen_weight = F.softmax(
                        chunk_top_value / self.direct_tau, dim=-1
                    )
                    chunk_screen_weight = (
                        chunk_screen_weight
                        * chunk_top_valid.to(chunk_screen_weight.dtype)
                    )
                    chunk_screen_weight = chunk_screen_weight / (
                        chunk_screen_weight.sum(dim=-1, keepdim=True)
                        .clamp_min(1e-8)
                    )
                    chunk_friend_local = torch.sum(
                        chunk_screen_weight.unsqueeze(-1)
                        * chunk_selected_value,
                        dim=-2,
                    )
                    chunk_user_consistency = torch.sum(
                        F.normalize(chunk_friend_local, dim=-1)
                        * local_axis.detach().unsqueeze(-2),
                        dim=-1,
                    )
                else:
                    chunk_user_consistency = torch.zeros_like(chunk_match)
                chunk_background = torch.einsum(
                    "bnk,bfk->bnf",
                    item_router_probability.detach(),
                    friend_pool_background[:, friend_start:friend_stop],
                )
                chunk_lift = chunk_match - chunk_background
                chunk_score = (
                    chunk_top_value.masked_fill(
                        ~chunk_top_valid, 0.0
                    ).sum(dim=-1)
                    / chunk_top_valid.sum(dim=-1).clamp_min(1)
                )
                if self.direct_calibrated_trust:
                    chunk_score = (
                        chunk_lift
                        + 0.20 * chunk_support
                        + 0.20
                        * torch.log(
                            friend_pool_trust[
                                :, friend_start:friend_stop
                            ].unsqueeze(1).clamp_min(1e-8)
                        )
                        + self.direct_prior_weight
                        * torch.log(
                            relative_prior[
                                :, friend_start:friend_stop
                            ].unsqueeze(1).clamp_min(1e-8)
                        )
                    )
                else:
                    if self.direct_user_consistent_screen:
                        chunk_score = (
                            chunk_score
                            + self.direct_screen_consistency_weight
                            * chunk_user_consistency
                            + 0.20 * chunk_support
                            + self.direct_screen_prior_weight
                            * torch.log(
                                relative_prior[
                                    :, friend_start:friend_stop
                                ].unsqueeze(1).clamp_min(1e-8)
                            )
                        )
                    else:
                        chunk_score = (
                            chunk_score
                            + 0.20 * chunk_support
                            + 0.10
                            * relative_prior[
                                :, friend_start:friend_stop
                            ].unsqueeze(1)
                        )
                chunk_score = chunk_score.masked_fill(
                    ~chunk_eligible, -1e4
                )
                score_parts.append(chunk_score)
                eligible_parts.append(chunk_eligible)
                support_parts.append(chunk_support)
                route_support_parts.append(chunk_route_support)
                match_parts.append(chunk_match)
                user_consistency_parts.append(chunk_user_consistency)
                history_index_parts.append(chunk_top_index_all)

            screen_score = torch.cat(score_parts, dim=2)
            screen_eligible = torch.cat(eligible_parts, dim=2)
            screen_support = torch.cat(support_parts, dim=2)
            screen_route_support = torch.cat(route_support_parts, dim=2)
            screen_match = torch.cat(match_parts, dim=2)
            screen_user_consistency = torch.cat(
                user_consistency_parts, dim=2
            )
            screen_top_index_all = torch.cat(history_index_parts, dim=2)
            selected_count = min(
                self.direct_friend_select, screen_score.size(-1)
            )
            selected_friend_index = torch.topk(
                screen_score, k=selected_count, dim=-1
            ).indices

        expanded_pool_ids = friend_pool_ids.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        expanded_pool_mask = friend_pool_mask.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        expanded_pool_prior = friend_pool_prior.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        selected_friend_ids = torch.gather(
            expanded_pool_ids, 2, selected_friend_index
        )
        selected_friend_mask = torch.gather(
            expanded_pool_mask, 2, selected_friend_index
        ) & torch.gather(screen_eligible, 2, selected_friend_index)
        selected_friend_prior = torch.gather(
            expanded_pool_prior, 2, selected_friend_index
        )
        selected_friend_trust = torch.gather(
            friend_pool_trust.unsqueeze(1).expand(-1, candidate_count, -1),
            2,
            selected_friend_index,
        )
        selected_screen_user_consistency = torch.gather(
            screen_user_consistency, 2, selected_friend_index
        )
        selected_friend_background = torch.gather(
            friend_pool_background.unsqueeze(1).expand(
                -1, candidate_count, -1, -1
            ),
            2,
            selected_friend_index.unsqueeze(-1).expand(
                -1, -1, -1, self.direct_router_interests
            ),
        )

        all_friend_history_ids = self.direct_friend_history_items[
            selected_friend_ids
        ]
        all_friend_history_valid = (
            self.direct_friend_history_mask[selected_friend_ids]
            & selected_friend_mask.unsqueeze(-1)
        )
        if self.direct_two_stage_retrieval:
            # Gather only each selected friend's candidate-specific history
            # shortlist into the differentiable branch.  The large
            # [friend_pool, history_pool] search above remains graph-free.
            selected_screen_history_index = torch.gather(
                screen_top_index_all,
                2,
                selected_friend_index.unsqueeze(-1).expand(
                    -1, -1, -1, screen_retrieve_count
                ),
            )
            history_short_count = min(
                self.direct_friend_topl,
                selected_screen_history_index.size(-1),
            )
            selected_screen_history_index = (
                selected_screen_history_index[..., :history_short_count]
            )
            friend_history_ids = torch.gather(
                all_friend_history_ids,
                -1,
                selected_screen_history_index,
            )
            friend_history_valid = torch.gather(
                all_friend_history_valid,
                -1,
                selected_screen_history_index,
            )
        else:
            friend_history_ids = all_friend_history_ids
            friend_history_valid = all_friend_history_valid
        friend_history_e = item_source[friend_history_ids]
        friend_router_probability = self._direct_router_probability(
            friend_history_e
        )
        friend_key = F.normalize(
            self.direct_key_proj(friend_history_e), dim=-1
        )
        friend_raw_similarity = torch.sum(
            query_unit.unsqueeze(-2).unsqueeze(-2) * friend_key, dim=-1
        )
        friend_route_compatibility = torch.sum(
            item_router_probability.unsqueeze(-2).unsqueeze(-2)
            * friend_router_probability,
            dim=-1,
        )
        friend_route_bias = torch.log(
            self.direct_router_interests * friend_route_compatibility + 1e-8
        )
        friend_similarity = (
            friend_raw_similarity
            + self.direct_router_weight * friend_route_bias
        )
        friend_similarity = friend_similarity.masked_fill(
            ~friend_history_valid, -1e4
        )
        friend_raw_similarity = friend_raw_similarity.masked_fill(
            ~friend_history_valid, -1e4
        )
        friend_top_count = min(
            self.direct_friend_topl, friend_similarity.size(-1)
        )
        friend_top_value, friend_top_index = torch.topk(
            friend_similarity, k=friend_top_count, dim=-1
        )
        friend_top_valid = torch.gather(
            friend_history_valid, -1, friend_top_index
        )
        friend_top_raw_value = torch.gather(
            friend_raw_similarity, -1, friend_top_index
        )
        friend_top_route = torch.gather(
            friend_route_compatibility, -1, friend_top_index
        )
        friend_history_weight = F.softmax(
            friend_top_value / self.direct_tau, dim=-1
        )
        friend_history_weight = (
            friend_history_weight
            * friend_top_valid.to(friend_history_weight.dtype)
        )
        friend_history_weight = friend_history_weight / friend_history_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        friend_route_support = torch.sum(
            friend_history_weight * friend_top_route, dim=-1
        )
        if self.direct_v1_only or self.direct_novelty_v1:
            # v1 predates the prototype router.  Keep its fixed eight-friend
            # raw-history path isolated from the later v4 routing signal.
            friend_route_support = torch.ones_like(friend_route_support)
        friend_value = self.direct_value_proj(friend_history_e)
        friend_selected = torch.gather(
            friend_value,
            3,
            friend_top_index.unsqueeze(-1).expand(
                -1, -1, -1, -1, dim
            ),
        )
        friend_local = torch.sum(
            friend_history_weight.unsqueeze(-1) * friend_selected, dim=-2
        )
        friend_available = friend_top_valid.any(dim=-1)
        match_probability = torch.sigmoid(
            (friend_top_raw_value - self.direct_match_threshold) / self.direct_tau
        ) * friend_top_valid.to(item_e.dtype)
        multi_support = match_probability.sum(dim=-1) / float(friend_top_count)
        friend_match = torch.sum(
            friend_history_weight * friend_top_raw_value, dim=-1
        )
        friend_background = torch.sum(
            item_router_probability.unsqueeze(-2) * selected_friend_background,
            dim=-1,
        )
        friend_lift = friend_match - friend_background
        friend_prior = (
            selected_friend_prior * friend_available.to(item_e.dtype)
        )
        friend_prior = friend_prior / friend_prior.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        friend_unit = F.normalize(friend_local, dim=-1)
        friend_projected = F.normalize(
            self.direct_friend_proj(friend_local), dim=-1
        )
        relevance = torch.sum(
            friend_projected * query_unit.unsqueeze(-2), dim=-1
        )
        consistency = torch.sum(
            friend_unit * local_axis.unsqueeze(-2), dim=-1
        )
        global_axis = F.normalize(user_anchor, dim=-1)
        local_global_consistency = torch.sum(
            local_axis * global_axis, dim=-1
        )
        friend_global_consistency = torch.sum(
            friend_unit * global_axis.unsqueeze(-2), dim=-1
        )
        local_global_reliability = torch.sigmoid(
            local_global_consistency / self.direct_tau
        )
        friend_local_reliability = torch.sigmoid(consistency / self.direct_tau)
        friend_consistency_reliability = torch.sqrt(
            friend_local_reliability
            * local_global_reliability.unsqueeze(-1)
        )
        parallel_scale = torch.sum(
            friend_local * local_axis.unsqueeze(-2), dim=-1, keepdim=True
        )
        friend_parallel = parallel_scale * local_axis.unsqueeze(-2)
        friend_orthogonal = friend_local - friend_parallel
        if self.direct_v1_only:
            gate_features = torch.stack(
                [
                    relevance,
                    consistency,
                    multi_support,
                    friend_prior,
                    user_item_cosine.unsqueeze(-1).expand_as(relevance),
                    local_global_consistency.unsqueeze(-1).expand_as(relevance),
                ],
                dim=-1,
            )
        else:
            gate_features = torch.stack(
                [
                    relevance,
                    consistency,
                    multi_support,
                    friend_prior,
                    friend_route_support,
                    user_item_cosine.unsqueeze(-1).expand_as(relevance),
                    local_global_consistency.unsqueeze(-1).expand_as(relevance),
                ],
                dim=-1,
            )
        friend_gate = torch.sigmoid(self.direct_gate_mlp(gate_features))
        if self.direct_novelty_v1:
            # The useful part of a candidate is not its complete direction,
            # but the direction that the user's retrieved local interest does
            # not already explain.  Project each friend's orthogonal evidence
            # onto that candidate-specific residual and discard negative or
            # unrelated components.  The 0.25 floor is safe here because the
            # retained vector is already both orthogonal to the user's local
            # interest and positively aligned with the candidate; the null
            # expert can still reject the whole friend.
            candidate_unexplained = (
                item_unit
                - user_item_cosine.unsqueeze(-1) * local_axis
            )
            candidate_unexplained_norm = candidate_unexplained.norm(
                dim=-1, keepdim=True
            )
            candidate_unexplained_axis = (
                candidate_unexplained
                / candidate_unexplained_norm.clamp_min(1e-8)
            )
            candidate_unexplained_axis = torch.where(
                candidate_unexplained_norm > 1e-6,
                candidate_unexplained_axis,
                torch.zeros_like(candidate_unexplained_axis),
            )
            novelty_coefficient = torch.sum(
                friend_orthogonal
                * candidate_unexplained_axis.unsqueeze(-2),
                dim=-1,
                keepdim=True,
            ).clamp_min(0.0)
            friend_candidate_novelty = (
                novelty_coefficient
                * candidate_unexplained_axis.unsqueeze(-2)
            )
            novelty_retention = (
                0.25 + 0.75 * friend_gate.mean(dim=-1, keepdim=True)
            )
            friend_clean = (
                friend_parallel
                + novelty_retention * friend_candidate_novelty
            )
            friend_novelty_support = (
                novelty_coefficient.squeeze(-1)
                / friend_orthogonal.norm(dim=-1).clamp_min(1e-8)
            ).clamp(0.0, 1.0)
        else:
            friend_clean = friend_parallel + friend_gate * friend_orthogonal
            friend_candidate_novelty = friend_gate * friend_orthogonal
            friend_novelty_support = torch.zeros_like(relevance)
        clean_item_cosine = torch.sum(
            F.normalize(friend_clean, dim=-1) * item_unit.unsqueeze(-2),
            dim=-1,
        )

        if self.direct_v1_only:
            evidence_features = torch.stack(
                [
                    relevance,
                    consistency,
                    clean_item_cosine,
                    multi_support,
                    friend_prior,
                    friend_global_consistency,
                ],
                dim=-1,
            )
        else:
            evidence_features = torch.stack(
                [
                    relevance,
                    consistency,
                    clean_item_cosine,
                    multi_support,
                    friend_prior,
                    friend_route_support,
                    friend_global_consistency,
                ],
                dim=-1,
            )
        evidence_residual = self.direct_residual_max * torch.tanh(
            self.direct_evidence_mlp(evidence_features).squeeze(-1)
        )
        evidence = (
            relevance
            + self.direct_consistency_weight * consistency
            + evidence_residual
        )
        if self.direct_calibrated_trust:
            # Trust is candidate-independent. Lift is candidate-specific and
            # zero-centered by the friend's normal same-route support.
            friend_logits = (
                (friend_lift - self.direct_lift_margin) / self.direct_tau
                + torch.log(selected_friend_trust.clamp_min(1e-8))
                + torch.log(multi_support.clamp_min(1e-8))
                + self.direct_prior_weight
                * torch.log(friend_prior.clamp_min(1e-8))
            )
            null_logit_value = 0.0
        else:
            friend_logits = evidence / self.direct_tau
            friend_logits = (
                friend_logits
                + torch.log(friend_prior.clamp_min(1e-8))
                + torch.log(multi_support.clamp_min(1e-8))
                + torch.log(friend_consistency_reliability.clamp_min(1e-8))
                + self.direct_router_weight * torch.log(
                    self.direct_router_interests * friend_route_support + 1e-8
                )
            )
            null_logit_value = self.direct_evidence_threshold / self.direct_tau
        friend_logits = friend_logits.masked_fill(~friend_available, -1e4)
        null_logit = torch.full_like(
            friend_logits[..., :1],
            null_logit_value,
        )
        choice_probability = F.softmax(
            torch.cat([friend_logits, null_logit], dim=-1), dim=-1
        )
        friend_weight = choice_probability[..., :-1]
        null_probability = choice_probability[..., -1]
        candidate_mass = friend_weight.sum(dim=-1)
        conditional_weight = friend_weight / candidate_mass.unsqueeze(
            -1
        ).clamp_min(1e-8)
        candidate_social = torch.sum(
            conditional_weight.unsqueeze(-1) * friend_clean, dim=-2
        )
        candidate_social_cosine = torch.sum(
            F.normalize(candidate_social, dim=-1) * item_unit, dim=-1
        )
        candidate_advantage = candidate_social_cosine - user_item_cosine
        candidate_scale = self.direct_scale_max * torch.sigmoid(
            self.direct_scale_logit
        )
        candidate_score = candidate_mass * candidate_scale * candidate_social_cosine
        if self.direct_calibrated_trust and not bool(
            self.direct_calibration_ready.item()
        ):
            # Epochs before the first train-only calibration are pure ODSR
            # warm-up. Keep the graph connected but suppress all branch output.
            candidate_score = candidate_score * 0.0
            candidate_mass = candidate_mass * 0.0
            null_probability = torch.ones_like(null_probability)
            friend_weight = friend_weight * 0.0
        weighted_relevance = torch.sum(
            conditional_weight * relevance, dim=-1
        )
        weighted_consistency = torch.sum(
            conditional_weight * consistency, dim=-1
        )
        weighted_support = torch.sum(
            conditional_weight * multi_support, dim=-1
        )
        weighted_route_support = torch.sum(
            conditional_weight * friend_route_support, dim=-1
        )
        weighted_lift = torch.sum(
            conditional_weight * friend_lift, dim=-1
        )
        weighted_trust = torch.sum(
            conditional_weight * selected_friend_trust, dim=-1
        )
        weighted_parallel = torch.sum(
            conditional_weight.unsqueeze(-1) * friend_parallel, dim=-2
        )
        weighted_orthogonal = torch.sum(
            conditional_weight.unsqueeze(-1) * friend_orthogonal, dim=-2
        )
        weighted_candidate_novelty = torch.sum(
            conditional_weight.unsqueeze(-1) * friend_candidate_novelty,
            dim=-2,
        )
        weighted_novelty_support = torch.sum(
            conditional_weight * friend_novelty_support, dim=-1
        )

        details = {
            "candidate_score": candidate_score,
            "candidate_scale": candidate_scale.expand_as(candidate_score),
            "candidate_mass": candidate_mass,
            "social_available": friend_available.any(dim=-1).to(item_e.dtype),
            "null_probability": null_probability,
            "friend_weight": friend_weight,
            "local_weight": own_weight,
            "own_selected_items": own_selected_ids,
            "relevance": relevance,
            "consistency": consistency,
            "soft_consistency": friend_consistency_reliability,
            "support": multi_support,
            "route_support": friend_route_support,
            "friend_lift": friend_lift,
            "friend_trust": selected_friend_trust,
            "evidence": evidence,
            "evidence_residual": evidence_residual,
            "candidate_parallel": weighted_parallel,
            "candidate_orthogonal": weighted_orthogonal,
            "candidate_novelty": weighted_candidate_novelty,
            "candidate_novelty_support": weighted_novelty_support,
            "candidate_gate": friend_gate.mean(dim=(-1, -2)),
            "candidate_clean": candidate_social,
            "candidate_cosine": candidate_social_cosine,
            "candidate_router_probability": item_router_probability,
            "candidate_social_cosine": candidate_social_cosine,
            "candidate_advantage": candidate_advantage,
            "user_item_cosine": user_item_cosine,
            "local_global_consistency": local_global_consistency,
            "weighted_relevance": weighted_relevance,
            "weighted_consistency": weighted_consistency,
            "weighted_support": weighted_support,
            "weighted_route_support": weighted_route_support,
            "weighted_lift": weighted_lift,
            "weighted_trust": weighted_trust,
            "own_relevance": own_relevance,
            "own_support": own_support,
            "friend_available_count": friend_available.to(item_e.dtype).sum(dim=-1),
            "friend_pool_available_count": screen_eligible.to(item_e.dtype).sum(dim=-1),
            "friend_pool_support_mean": screen_support.mean(dim=-1),
            "friend_pool_route_mean": screen_route_support.mean(dim=-1),
            "friend_screen_user_consistency": (
                selected_screen_user_consistency
                * selected_friend_mask.to(item_e.dtype)
            ).sum(dim=-1) / selected_friend_mask.sum(dim=-1).clamp_min(1),
            "raw_delta": candidate_score,
        }
        if squeeze_candidate:
            candidate_score = candidate_score[:, 0]
            details = {
                key: (value[:, 0] if value.dim() >= 2 and value.size(1) == 1 else value)
                for key, value in details.items()
            }
        return candidate_score, details

    def _candidate_correction(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        final_item: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        detach_backbone: bool = False,
        base_score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.candidate_direct_history_expert:
            return self._direct_history_correction(
                users, items, final_item, aux, detach_backbone
            )
        if not self.candidate_retrieval_expert:
            return super()._candidate_correction(
                users, items, final_item, aux, detach_backbone, base_score
            )

        item_e = final_item[items]
        own_interest = self._candidate_user_view(
            aux["retrieval_own_interest"], users, item_e
        )
        own_mass = self._candidate_user_view(
            aux["retrieval_own_mass"], users, item_e
        )
        social_interest = self._candidate_user_view(
            aux["retrieval_social_interest"], users, item_e
        )
        social_mass = self._candidate_user_view(
            aux["retrieval_social_mass"], users, item_e
        )
        user_anchor = self._candidate_user_view(
            aux["retrieval_user_anchor"], users, item_e
        )
        if self.retrieval_specificity:
            own_dispersion = self._candidate_user_view(
                aux["retrieval_own_dispersion"], users, item_e
            )
            social_dispersion = self._candidate_user_view(
                aux["retrieval_social_dispersion"], users, item_e
            )
        else:
            own_dispersion = None
            social_dispersion = None
        if detach_backbone:
            item_e = item_e.detach()
            own_interest = own_interest.detach()
            own_mass = own_mass.detach()
            social_interest = social_interest.detach()
            social_mass = social_mass.detach()
            user_anchor = user_anchor.detach()
            if own_dispersion is not None:
                own_dispersion = own_dispersion.detach()
                social_dispersion = social_dispersion.detach()

        item_unit = F.normalize(item_e, dim=-1)
        query_unit = F.normalize(self.retrieval_query_proj(item_e), dim=-1)

        # Candidate-local self interest: retrieve the matching portions of the
        # user's own history.  This is used only to validate social evidence and
        # is not itself added as a recommendation score.
        own_projected = F.normalize(
            self.retrieval_local_proj(own_interest), dim=-1
        )
        own_relevance = torch.sum(
            own_projected * query_unit.unsqueeze(-2), dim=-1
        )
        own_support = (own_mass / (own_mass + 1.0)).expand_as(own_relevance)
        own_valid = (own_mass > 1e-8).expand_as(own_relevance)
        if own_dispersion is None:
            own_compactness = torch.ones_like(own_relevance)
        else:
            own_compactness = torch.exp(
                -self.retrieval_dispersion_weight * own_dispersion
            ).expand_as(own_relevance)
        own_logits = (
            own_relevance / self.retrieval_tau
            + torch.log(own_support.clamp_min(1e-8))
            + torch.log(own_compactness.clamp_min(1e-8))
        )
        own_logits = own_logits.masked_fill(~own_valid, -1e4)
        local_weight = F.softmax(own_logits, dim=-1) * own_valid.to(own_logits.dtype)
        local_weight = local_weight / local_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        local_interest = torch.sum(
            local_weight.unsqueeze(-1) * own_interest, dim=-2
        )
        own_available = own_valid.any(dim=-1)
        local_interest = torch.where(
            own_available.unsqueeze(-1), local_interest, user_anchor
        )
        local_axis = F.normalize(local_interest, dim=-1)
        user_item_cosine = torch.sum(local_axis * item_unit, dim=-1)
        masked_own_relevance = own_relevance.masked_fill(~own_valid, -1e4)
        best_own_relevance = masked_own_relevance.max(dim=-1).values
        best_own_relevance = torch.where(
            own_available, best_own_relevance, user_item_cosine
        )

        social_projected = F.normalize(
            self.retrieval_proto_proj(social_interest), dim=-1
        )
        relevance = torch.sum(
            social_projected * query_unit.unsqueeze(-2), dim=-1
        )
        consistency = torch.sum(
            F.normalize(social_interest, dim=-1)
            * local_axis.unsqueeze(-2),
            dim=-1,
        )
        support = (social_mass / (social_mass + 1.0)).expand_as(relevance)
        social_valid = (social_mass > 1e-8).expand_as(relevance)
        if social_dispersion is None:
            prototype_compactness = torch.ones_like(relevance)
            social_dispersion = torch.zeros_like(relevance)
        else:
            social_dispersion = social_dispersion.expand_as(relevance)
            prototype_compactness = torch.exp(
                -self.retrieval_dispersion_weight * social_dispersion
            )

        evidence_features = torch.stack(
            [
                relevance,
                consistency,
                support,
                user_item_cosine.unsqueeze(-1).expand_as(relevance),
                best_own_relevance.unsqueeze(-1).expand_as(relevance),
            ],
            dim=-1,
        )
        evidence_residual = self.retrieval_residual_max * torch.tanh(
            self.retrieval_evidence_mlp(evidence_features).squeeze(-1)
        )
        evidence = (
            relevance
            + self.retrieval_consistency_weight * consistency
            + evidence_residual
        )
        prototype_logits = (
            evidence / self.retrieval_tau
            + torch.log(support.clamp_min(1e-8))
            + torch.log(prototype_compactness.clamp_min(1e-8))
        )
        prototype_logits = prototype_logits.masked_fill(~social_valid, -1e4)
        null_logit = torch.full_like(
            prototype_logits[..., :1],
            self.retrieval_evidence_threshold / self.retrieval_tau
            + math.log(float(self.retrieval_num_interests)),
        )
        choice_probability = F.softmax(
            torch.cat([prototype_logits, null_logit], dim=-1), dim=-1
        )
        prototype_weight = choice_probability[..., :-1]
        null_probability = choice_probability[..., -1]
        candidate_mass = prototype_weight.sum(dim=-1)
        conditional_weight = prototype_weight / candidate_mass.unsqueeze(-1).clamp_min(1e-8)
        retrieved_social = torch.sum(
            conditional_weight.unsqueeze(-1) * social_interest, dim=-2
        )

        # ODSR-style decomposition is performed after retrieval.  Parallel
        # evidence is kept; only the candidate-local orthogonal part is gated.
        parallel_scale = torch.sum(
            retrieved_social * local_axis, dim=-1, keepdim=True
        )
        candidate_parallel = parallel_scale * local_axis
        candidate_orthogonal = retrieved_social - candidate_parallel
        retrieved_item_cosine = torch.sum(
            F.normalize(retrieved_social, dim=-1) * item_unit, dim=-1
        )
        weighted_relevance = torch.sum(conditional_weight * relevance, dim=-1)
        weighted_consistency = torch.sum(conditional_weight * consistency, dim=-1)
        weighted_support = torch.sum(conditional_weight * support, dim=-1)
        weighted_dispersion = torch.sum(
            conditional_weight * social_dispersion, dim=-1
        )
        weighted_compactness = torch.sum(
            conditional_weight * prototype_compactness, dim=-1
        )
        gate_features = torch.stack(
            [
                user_item_cosine,
                weighted_relevance,
                weighted_consistency,
                weighted_support,
                candidate_mass,
                retrieved_item_cosine,
            ],
            dim=-1,
        )
        candidate_gate = torch.sigmoid(self.retrieval_gate_mlp(gate_features))
        candidate_clean = candidate_parallel + candidate_gate * candidate_orthogonal
        candidate_cosine = torch.sum(
            F.normalize(candidate_clean, dim=-1) * item_unit, dim=-1
        )
        candidate_scale = self.retrieval_scale_max * torch.sigmoid(
            self.retrieval_scale_logit
        )
        candidate_score = candidate_mass * candidate_scale * candidate_cosine

        details = {
            "candidate_score": candidate_score,
            "candidate_scale": candidate_scale.expand_as(candidate_score),
            "candidate_mass": candidate_mass,
            "social_available": social_valid.any(dim=-1).to(candidate_score.dtype),
            "null_probability": null_probability,
            "prototype_weight": prototype_weight,
            "local_weight": local_weight,
            "relevance": relevance,
            "consistency": consistency,
            "support": support,
            "prototype_dispersion": social_dispersion,
            "prototype_compactness": prototype_compactness,
            "evidence": evidence,
            "evidence_residual": evidence_residual,
            "retrieved_social": retrieved_social,
            "candidate_parallel": candidate_parallel,
            "candidate_orthogonal": candidate_orthogonal,
            "candidate_gate": candidate_gate,
            "candidate_clean": candidate_clean,
            "candidate_cosine": candidate_cosine,
            "weighted_relevance": weighted_relevance,
            "weighted_consistency": weighted_consistency,
            "weighted_support": weighted_support,
            "weighted_dispersion": weighted_dispersion,
            "weighted_compactness": weighted_compactness,
            "own_relevance": own_relevance,
            "raw_delta": candidate_score,
        }
        # This is an independent bounded score from newly retrieved social
        # evidence.  The historical method name is kept only for API stability.
        return candidate_score, details

    def _compute_direct_history_loss(
        self,
        users: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        candidate_pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if neg.dim() == 1:
            neg = neg.unsqueeze(1)
        final_user, final_item, aux = self._backbone(
            current_ssl_mode=None, return_aux=True
        )
        pos_score, pos_details = self.score_pairs(
            users, pos, final_user, final_item, aux
        )
        neg_score, neg_details = self.score_pairs(
            users, neg, final_user, final_item, aux
        )
        main_bpr = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()

        u_e = final_user[users]
        p_e = final_item[pos]
        n_e = final_item[neg]
        reg_loss = self.reg * (u_e.norm(2) + p_e.norm(2) + n_e.norm(2))

        if candidate_pool is None:
            pool = torch.randint(
                0,
                self.item_num,
                (users.size(0), self.hard_pool_size),
                device=users.device,
            )
        else:
            if (
                candidate_pool.dim() != 2
                or candidate_pool.size(0) != users.size(0)
                or candidate_pool.size(1) < self.hard_topk
            ):
                raise ValueError("candidate_pool must be [batch, pool_size>=hard_topk]")
            pool = candidate_pool.to(device=users.device, dtype=torch.long)
        with torch.no_grad():
            pool_base = torch.sum(
                final_user[users].unsqueeze(1) * final_item[pool], dim=-1
            )
            hard_count = min(self.hard_topk, pool.size(1))
            hard_index = torch.topk(pool_base, k=hard_count, dim=1).indices
            hard_items = torch.gather(pool, 1, hard_index)
            hard_base = torch.gather(pool_base, 1, hard_index)
            positive_base = torch.sum(
                final_user[users] * final_item[pos], dim=-1
            )
            # The standard deviation of each user's ODSR scores provides a
            # dataset- and user-specific unit for a ranking margin.  This
            # avoids asking Epinions and Douban to learn the same absolute
            # correction scale even though their dot-product scales differ.
            base_score_scale = pool_base.std(dim=1, unbiased=False).clamp_min(
                self.direct_score_scale_floor
            )
            normalized_base_margin = (
                positive_base.unsqueeze(1) - hard_base
            ) / base_score_scale.unsqueeze(1)

        pos_candidate, pos_aux = self._direct_history_correction(
            users, pos, final_item, aux, detach_backbone=True
        )
        hard_candidate, hard_aux = self._direct_history_correction(
            users, hard_items, final_item, aux, detach_backbone=True
        )
        candidate_margin = pos_candidate.unsqueeze(1) - hard_candidate
        normalized_candidate_margin = candidate_margin / base_score_scale.unsqueeze(1)
        if self.direct_rank_calibrated:
            # Only fill the margin that the current ODSR backbone is missing.
            # Once ODSR already separates a pair by target_margin_ratio, the
            # desired candidate correction is zero.  Smooth-L1 has a finite
            # optimum, unlike softplus(-margin), which keeps increasing the
            # global branch scale after validation ranking has peaked.
            target_candidate_margin = (
                self.direct_target_margin_ratio - normalized_base_margin
            ).clamp(min=0.0, max=self.direct_max_margin_ratio)
            pairwise_candidate_loss = F.smooth_l1_loss(
                normalized_candidate_margin,
                target_candidate_margin,
                beta=self.direct_rank_beta,
            )
            pointwise_candidate_loss = torch.zeros((), device=self.device)
            correction_energy = 0.5 * (
                (pos_candidate / base_score_scale).square().mean()
                + (
                    hard_candidate / base_score_scale.unsqueeze(1)
                ).square().mean()
            )
        else:
            target_candidate_margin = torch.zeros_like(
                normalized_candidate_margin
            )
            pairwise_candidate_loss = F.softplus(-candidate_margin).mean()
            pointwise_candidate_loss = 0.5 * (
                F.softplus(-pos_candidate).mean()
                + F.softplus(hard_candidate).mean()
            )
            correction_energy = torch.zeros((), device=self.device)
        pos_help_target = torch.sigmoid(
            pos_aux["candidate_cosine"].detach() / 0.1
        )
        hard_help_target = torch.sigmoid(
            -hard_aux["candidate_cosine"].detach() / 0.1
        )
        confidence_loss = 0.5 * (
            F.binary_cross_entropy(
                pos_aux["candidate_mass"].clamp(1e-6, 1.0 - 1e-6),
                pos_help_target,
            )
            + F.binary_cross_entropy(
                hard_aux["candidate_mass"].clamp(1e-6, 1.0 - 1e-6),
                hard_help_target,
            )
        )
        candidate_loss = (
            pairwise_candidate_loss
            + 0.25 * pointwise_candidate_loss
            + self.direct_confidence_weight * confidence_loss
            + (
                self.direct_correction_reg_weight * correction_energy
                if self.direct_rank_calibrated
                else 0.0
            )
        )
        if self.direct_v1_only:
            router_usage = torch.ones(1, device=self.device)
            router_balance_loss = torch.zeros((), device=self.device)
            router_entropy_loss = torch.zeros((), device=self.device)
            router_regularization = torch.zeros((), device=self.device)
        else:
            router_dim = pos_aux["candidate_router_probability"].size(-1)
            all_router_probability = torch.cat(
                [
                    pos_aux["candidate_router_probability"].reshape(-1, router_dim),
                    hard_aux["candidate_router_probability"].reshape(-1, router_dim),
                ],
                dim=0,
            )
            router_usage = all_router_probability.mean(dim=0)
            router_balance_loss = (
                router_usage - 1.0 / float(router_dim)
            ).square().sum()
            router_entropy_loss = -(
                all_router_probability
                * torch.log(all_router_probability.clamp_min(1e-8))
            ).sum(dim=-1).mean()
            router_regularization = (
                self.direct_router_balance_weight * router_balance_loss
                + self.direct_router_entropy_weight * router_entropy_loss
            )

        all_residual = torch.cat(
            [
                pos_details["evidence_residual"].reshape(-1),
                neg_details["evidence_residual"].reshape(-1),
            ]
        )
        gate_loss = self.direct_reg_weight * all_residual.square().mean()

        ssl_loss = torch.tensor(0.0, device=self.device)
        if self.ssl_reg > 0:
            v1_u, _ = self.forward(current_ssl_mode="adaptive")
            v2_u, _ = self.forward(current_ssl_mode=None)
            v1 = F.normalize(v1_u[users], dim=1)
            v2 = F.normalize(v2_u[users], dim=1)
            pos_sc = torch.exp((v1 * v2).sum(1) / self.ssl_temp)
            ttl_sc = torch.matmul(v1, v2.t()).div(self.ssl_temp).exp().sum(1)
            ssl_loss = self.ssl_reg * (-torch.log(pos_sc / ttl_sc).mean())

        total = (
            main_bpr
            + self.candidate_loss_weight * candidate_loss
            + ssl_loss
            + reg_loss
            + gate_loss
            + router_regularization
        )
        with torch.no_grad():
            all_mass = torch.cat(
                [
                    pos_details["candidate_mass"].reshape(-1),
                    neg_details["candidate_mass"].reshape(-1),
                ]
            )
            all_null = torch.cat(
                [
                    pos_details["null_probability"].reshape(-1),
                    neg_details["null_probability"].reshape(-1),
                ]
            )
            all_gate = torch.cat(
                [
                    pos_details["candidate_gate"].reshape(-1),
                    neg_details["candidate_gate"].reshape(-1),
                ]
            )
            all_available = torch.cat(
                [
                    pos_details["social_available"].reshape(-1),
                    neg_details["social_available"].reshape(-1),
                ]
            )
            self._last_loss_terms = {
                "main_bpr": float(main_bpr.item()),
                "candidate": float(candidate_loss.item()),
                "candidate_pairwise": float(pairwise_candidate_loss.item()),
                "candidate_confidence": float(confidence_loss.item()),
                "candidate_correction_reg": float(correction_energy.item()),
                "ssl": float(ssl_loss.item()),
                "reg": float(reg_loss.item()),
                "gate": float(gate_loss.item()),
                "router_balance": float(router_balance_loss.item()),
                "router_entropy": float(router_entropy_loss.item()),
                "router_reg": float(router_regularization.item()),
            }
            self._last_candidate_stats = {
                "correction_pos_mean": float(pos_candidate.mean().item()),
                "correction_hard_mean": float(hard_candidate.mean().item()),
                "direct_scale": float(pos_aux["candidate_scale"].mean().item()),
                "candidate_mass_mean": float(all_mass.mean().item()),
                "candidate_mass_pos": float(pos_aux["candidate_mass"].mean().item()),
                "candidate_mass_hard": float(hard_aux["candidate_mass"].mean().item()),
                "null_probability_mean": float(all_null.mean().item()),
                "social_available_frac": float(all_available.mean().item()),
                "direct_score_pos": float(pos_candidate.mean().item()),
                "direct_score_hard": float(hard_candidate.mean().item()),
                "candidate_margin": float(candidate_margin.mean().item()),
                "candidate_margin_normalized": float(
                    normalized_candidate_margin.mean().item()
                ),
                "candidate_target_margin_normalized": float(
                    target_candidate_margin.mean().item()
                ),
                "odsr_margin_normalized": float(
                    normalized_base_margin.mean().item()
                ),
                "odsr_deficit_fraction": float(
                    (target_candidate_margin > 0)
                    .to(pos_candidate.dtype)
                    .mean()
                    .item()
                ),
                "correction_energy": float(correction_energy.item()),
                "base_score_scale": float(base_score_scale.mean().item()),
                "relevance_pos": float(pos_details["weighted_relevance"].mean().item()),
                "relevance_hard": float(hard_aux["weighted_relevance"].mean().item()),
                "consistency_pos": float(pos_details["weighted_consistency"].mean().item()),
                "consistency_hard": float(hard_aux["weighted_consistency"].mean().item()),
                "candidate_gate_mean": float(all_gate.mean().item()),
                "own_support_pos": float(pos_details["own_support"].mean().item()),
                "friend_count_mean": float(pos_details["friend_available_count"].mean().item()),
                "friend_pool_available_mean": float(pos_details["friend_pool_available_count"].mean().item()),
                "friend_pool_support_mean": float(pos_details["friend_pool_support_mean"].mean().item()),
                "multi_support_pos": float(pos_details["weighted_support"].mean().item()),
                "local_global_consistency_pos": float(pos_details["local_global_consistency"].mean().item()),
                "social_cosine_pos": float(pos_details["candidate_social_cosine"].mean().item()),
                "social_advantage_pos": float(pos_details["candidate_advantage"].mean().item()),
                "self_cosine_pos": float(pos_details["user_item_cosine"].mean().item()),
                "candidate_novelty_support_pos": float(
                    pos_details["candidate_novelty_support"].mean().item()
                ) if "candidate_novelty_support" in pos_details else 0.0,
                "candidate_novelty_norm_pos": float(
                    pos_details["candidate_novelty"].norm(dim=-1).mean().item()
                ) if "candidate_novelty" in pos_details else 0.0,
                "route_support_pos": float(pos_details["weighted_route_support"].mean().item()),
                "route_support_hard": float(hard_aux["weighted_route_support"].mean().item()),
                "friend_lift_pos": float(pos_details["weighted_lift"].mean().item()),
                "friend_lift_hard": float(hard_aux["weighted_lift"].mean().item()),
                "friend_trust_pos": float(pos_details["weighted_trust"].mean().item()),
                "friend_trust_hard": float(hard_aux["weighted_trust"].mean().item()),
                "calibration_ready": float(self.direct_calibration_ready.item()),
                "router_usage_min": float(router_usage.min().item()),
                "router_usage_max": float(router_usage.max().item()),
                "router_entropy": float(router_entropy_loss.item()),
            }
        return total

    def _compute_candidate_loss(
        self,
        users: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        candidate_pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.candidate_direct_history_expert:
            return self._compute_direct_history_loss(
                users, pos, neg, candidate_pool
            )
        if not self.candidate_retrieval_expert:
            return super()._compute_candidate_loss(users, pos, neg, candidate_pool)
        if neg.dim() == 1:
            neg = neg.unsqueeze(1)

        final_user, final_item, aux = self._backbone(
            current_ssl_mode=None, return_aux=True
        )
        pos_score, pos_details = self.score_pairs(
            users, pos, final_user, final_item, aux
        )
        neg_score, neg_details = self.score_pairs(
            users, neg, final_user, final_item, aux
        )
        main_bpr = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()

        u_e = final_user[users]
        p_e = final_item[pos]
        n_e = final_item[neg]
        reg_loss = self.reg * (u_e.norm(2) + p_e.norm(2) + n_e.norm(2))

        if candidate_pool is None:
            pool = torch.randint(
                0,
                self.item_num,
                (users.size(0), self.hard_pool_size),
                device=users.device,
            )
        else:
            if (
                candidate_pool.dim() != 2
                or candidate_pool.size(0) != users.size(0)
                or candidate_pool.size(1) < self.hard_topk
            ):
                raise ValueError("candidate_pool must be [batch, pool_size>=hard_topk]")
            pool = candidate_pool.to(device=users.device, dtype=torch.long)
        with torch.no_grad():
            pool_base = torch.sum(
                final_user[users].unsqueeze(1) * final_item[pool], dim=-1
            )
            hard_count = min(self.hard_topk, pool.size(1))
            hard_idx = torch.topk(pool_base, k=hard_count, dim=1).indices
            hard_items = torch.gather(pool, 1, hard_idx)

        pos_candidate, pos_aux = self._candidate_correction(
            users,
            pos,
            final_item,
            aux,
            detach_backbone=True,
            base_score=(final_user[users] * final_item[pos]).sum(dim=-1),
        )
        hard_candidate, hard_aux = self._candidate_correction(
            users,
            hard_items,
            final_item,
            aux,
            detach_backbone=True,
            base_score=(
                final_user[users].unsqueeze(1) * final_item[hard_items]
            ).sum(dim=-1),
        )
        candidate_margin = pos_candidate.unsqueeze(1) - hard_candidate
        pairwise_candidate_loss = F.softplus(-candidate_margin).mean()
        pointwise_candidate_loss = 0.5 * (
            F.softplus(-pos_candidate).mean() + F.softplus(hard_candidate).mean()
        )

        # Calibrate the null option: use the branch only when its cleaned social
        # direction has the correct sign for the observed training pair.
        pos_help_target = torch.sigmoid(
            pos_aux["candidate_cosine"].detach() / 0.1
        )
        hard_help_target = torch.sigmoid(
            -hard_aux["candidate_cosine"].detach() / 0.1
        )
        confidence_loss = 0.5 * (
            F.binary_cross_entropy(
                pos_aux["candidate_mass"].clamp(1e-6, 1.0 - 1e-6),
                pos_help_target,
            )
            + F.binary_cross_entropy(
                hard_aux["candidate_mass"].clamp(1e-6, 1.0 - 1e-6),
                hard_help_target,
            )
        )
        candidate_mass_gap = (
            pos_aux["candidate_mass"].unsqueeze(1)
            - hard_aux["candidate_mass"]
        )
        if self.retrieval_specificity:
            mass_margin_loss = F.relu(
                self.retrieval_mass_margin - candidate_mass_gap
            ).mean()
        else:
            mass_margin_loss = torch.zeros((), device=users.device)
        candidate_loss = (
            pairwise_candidate_loss
            + 0.25 * pointwise_candidate_loss
            + self.retrieval_confidence_weight * confidence_loss
            + self.retrieval_mass_margin_weight * mass_margin_loss
        )

        evidence_residual = torch.cat(
            [
                pos_details["evidence_residual"].reshape(-1),
                neg_details["evidence_residual"].reshape(-1),
            ]
        )
        head_usage = aux["retrieval_head_usage"]
        balanced_usage = torch.full_like(
            head_usage, 1.0 / self.retrieval_num_interests
        )
        head_probability = aux["retrieval_head_probability"]
        assignment_entropy = -(
            head_probability * head_probability.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        gate_loss = (
            self.retrieval_reg_weight * evidence_residual.square().mean()
            + self.retrieval_balance_weight
            * (head_usage - balanced_usage).square().mean()
            + self.retrieval_specialization_weight * assignment_entropy
        )

        ssl_loss = torch.tensor(0.0, device=self.device)
        if self.ssl_reg > 0:
            v1_u, _ = self.forward(current_ssl_mode="adaptive")
            v2_u, _ = self.forward(current_ssl_mode=None)
            v1 = F.normalize(v1_u[users], dim=1)
            v2 = F.normalize(v2_u[users], dim=1)
            pos_sc = torch.exp((v1 * v2).sum(1) / self.ssl_temp)
            ttl_sc = torch.matmul(v1, v2.t()).div(self.ssl_temp).exp().sum(1)
            ssl_loss = self.ssl_reg * (-torch.log(pos_sc / ttl_sc).mean())

        total = (
            main_bpr
            + self.candidate_loss_weight * candidate_loss
            + ssl_loss
            + reg_loss
            + gate_loss
        )
        with torch.no_grad():
            all_mass = torch.cat(
                [
                    pos_details["candidate_mass"].reshape(-1),
                    neg_details["candidate_mass"].reshape(-1),
                ]
            )
            all_null = torch.cat(
                [
                    pos_details["null_probability"].reshape(-1),
                    neg_details["null_probability"].reshape(-1),
                ]
            )
            all_gate = torch.cat(
                [
                    pos_details["candidate_gate"].reshape(-1),
                    neg_details["candidate_gate"].reshape(-1),
                ]
            )
            all_available = torch.cat(
                [
                    pos_details["social_available"].reshape(-1),
                    neg_details["social_available"].reshape(-1),
                ]
            )
            self._last_loss_terms = {
                "main_bpr": float(main_bpr.item()),
                "candidate": float(candidate_loss.item()),
                "candidate_pairwise": float(pairwise_candidate_loss.item()),
                "candidate_confidence": float(confidence_loss.item()),
                "candidate_mass_margin": float(mass_margin_loss.item()),
                "ssl": float(ssl_loss.item()),
                "reg": float(reg_loss.item()),
                "gate": float(gate_loss.item()),
            }
            self._last_candidate_stats = {
                "correction_pos_mean": float(pos_candidate.mean().item()),
                "correction_hard_mean": float(hard_candidate.mean().item()),
                "retrieval_scale": float(pos_aux["candidate_scale"].mean().item()),
                "candidate_mass_mean": float(all_mass.mean().item()),
                "candidate_mass_pos": float(pos_aux["candidate_mass"].mean().item()),
                "candidate_mass_hard": float(hard_aux["candidate_mass"].mean().item()),
                "null_probability_mean": float(all_null.mean().item()),
                "social_available_frac": float(all_available.mean().item()),
                "new_social_score_pos": float(pos_candidate.mean().item()),
                "new_social_score_hard": float(hard_candidate.mean().item()),
                "candidate_margin": float(candidate_margin.mean().item()),
                "candidate_mass_gap": float(candidate_mass_gap.mean().item()),
                "prototype_dispersion_pos": float(
                    pos_aux["weighted_dispersion"].mean().item()
                ),
                "prototype_dispersion_hard": float(
                    hard_aux["weighted_dispersion"].mean().item()
                ),
                "prototype_compactness_pos": float(
                    pos_aux["weighted_compactness"].mean().item()
                ),
                "prototype_compactness_hard": float(
                    hard_aux["weighted_compactness"].mean().item()
                ),
                "relevance_pos": float(pos_details["weighted_relevance"].mean().item()),
                "relevance_hard": float(hard_aux["weighted_relevance"].mean().item()),
                "consistency_pos": float(pos_details["weighted_consistency"].mean().item()),
                "consistency_hard": float(hard_aux["weighted_consistency"].mean().item()),
                "candidate_gate_mean": float(all_gate.mean().item()),
                "head_usage_min": float(head_usage.min().item()),
                "head_usage_max": float(head_usage.max().item()),
                "assignment_entropy": float(assignment_entropy.item()),
            }
        return total
