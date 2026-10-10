"""Source compatibility base for the preserved-reference ODSR subclass.

The reference server loaded this API from a frozen ``.pyc``.  This module
adapts the readable local ODSR backbone to that API without changing the
candidate-conditioned direct-history implementation.
"""
from typing import Dict, Optional, Tuple

import torch

import ODSR_legacy as _legacy


ODSRLayer = _legacy.ODSRLayer


class ODSR(_legacy.ODSR):
    """Expose the newer candidate-expert hooks over the local backbone."""

    def set_training_epoch(self, epoch: int) -> None:
        """Expose the epoch hook expected by the reference expert."""
        self.current_epoch = int(epoch)

    def _backbone(
        self,
        current_ssl_mode: Optional[str] = None,
        return_aux: bool = False,
    ):
        if not return_aux:
            return self.encode(
                current_ssl_mode=current_ssl_mode,
                return_context=False,
            )
        final_user, final_item, context = self.encode(
            current_ssl_mode=current_ssl_mode,
            return_context=True,
        )
        aux = dict(context)
        aux["retrieval_user_anchor"] = final_user
        return final_user, final_item, aux

    def _candidate_correction(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        final_item: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        detach_backbone: bool = False,
        base_score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        item_e = final_item[items]
        score = item_e.new_zeros(item_e.shape[:-1])
        return score, {"raw_delta": score}

    def _compute_candidate_loss(
        self,
        users: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        candidate_pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("candidate subclass must implement its loss")

    def score_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        final_user: Optional[torch.Tensor] = None,
        final_item: Optional[torch.Tensor] = None,
        aux: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if final_user is None or final_item is None or aux is None:
            final_user, final_item, aux = self._backbone(
                current_ssl_mode=None,
                return_aux=True,
            )
        item_e = final_item[items]
        user_e = final_user[users]
        if item_e.dim() == 3:
            user_e = user_e.unsqueeze(1)
        base_score = torch.sum(user_e * item_e, dim=-1)
        correction, details = self._candidate_correction(
            users,
            items,
            final_item,
            aux,
            detach_backbone=False,
            base_score=base_score,
        )
        return base_score + correction, details

    def compute_loss(
        self,
        users: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        backbone_neg: Optional[torch.Tensor] = None,
        candidate_pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if neg.dim() == 1:
            neg = neg.unsqueeze(1)
        loss = self._compute_candidate_loss(
            users,
            pos,
            neg,
            candidate_pool=candidate_pool,
        )
        terms = getattr(self, "_last_loss_terms", {})
        self.last_rec_components = {
            "fused": loss.detach().new_tensor(
                float(terms.get("main_bpr", loss.detach().item()))
            ),
            "candidate": loss.detach().new_tensor(
                float(terms.get("candidate", 0.0))
            ),
        }
        return loss

    @torch.no_grad()
    def score_all_items(
        self,
        users: torch.Tensor,
        context: Dict[str, torch.Tensor],
        item_batch: int = 512,
    ) -> torch.Tensor:
        aux = dict(context)
        aux["retrieval_user_anchor"] = context["final_user"]
        return self.score_matrix(
            users,
            final_user=context["final_user"],
            final_item=context["final_item"],
            aux=aux,
            item_chunk=item_batch,
        )
