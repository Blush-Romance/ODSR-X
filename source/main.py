# main.py
# -*- coding: utf-8 -*-
import argparse
import os
import time
from typing import Dict, Any, Set, Tuple

import numpy as np
import torch
import torch.optim as optim

from common import GraphData, seed_everything
from ODSR import ODSR
from utils import evaluate_full_ranking


COMMON: Dict[str, Any] = dict(
    seed=2026,
    lr=0.001,
    emb_size=64,
    n_layer=3,
    tau=0.20,
    batch_size=1024,
    eval_interval=1,
    patience=50,
    eval_batch=64,
    eval_item_batch=512,
    item_aware_mode="orth_gate",
    negative_k=8,
    negative_strategy="random_mean",
    negative_gradient_mode="joint",
    candidate_loss_weight=0.2,
    hard_candidate_pool=64,
    candidate_residual_bound=0.25,
    candidate_recovery_scale=0.1,
    candidate_confidence_bias=-4.0,
    signed_gate_scale=0.2,
    gate_change_reg_weight=0.0,
    gate_saturation_bound=3.0,
    gate_saturation_reg_weight=0.0,
    confidence_warmup_epochs=0,
    proposal_warmup_loss_weight=0.0,
    proposal_aux_loss_weight=0.0,
    item_residual_bound=0.1,
    conditional_residual_bound=0.1,
    normalized_cross_scale=0.05,
    item_cosine_groups=4,
    item_branch_loss_weight=0.0,
    conditional_branch_loss_weight=0.0,
    branch_aux_score_mode="full",
    init_ckpt=None,
    finetune_role="scratch",
    freeze_backbone_epochs=0,
    backbone_lr_scale=0.1,
    metric="recall@20",
    candidate_direct_history_expert=True,
    candidate_direct_v1_only=True,
    candidate_direct_user_history_max=64,
    candidate_direct_friend_max=8,
    candidate_direct_friend_select=8,
    candidate_direct_friend_history_max=16,
    candidate_direct_user_topl=8,
    candidate_direct_friend_topl=4,
    candidate_direct_eval_topm=512,
    candidate_direct_tau=0.2,
    candidate_direct_consistency_weight=1.0,
    candidate_direct_reg=1e-4,
    candidate_direct_confidence_weight=0.05,
    hard_pool_size=64,
    hard_topk=8,
)

DATASET_CFG: Dict[str, Dict[str, Any]] = {
    "douban-book": dict(reg=0.00, ssl_reg=0.58, eps=0.57, dropout=0.01, max_epoch=60),
    "epinions": dict(reg=0.00, ssl_reg=0.18, eps=0.46, dropout=0.14, max_epoch=30),
    "yelp2018": dict(reg=0.00, ssl_reg=0.43, eps=0.54, dropout=0.17, max_epoch=20),
}

CAP_CKPT = {
    "douban-book": "Douban-Book.pth",
    "epinions": "Epinions.pth",
    "yelp2018": "Yelp2018.pth",
}
LOW_CKPT = {
    "douban-book": "douban-book-candidate-delta-residual-bounded025-k8.pth",
    "epinions": "epinions-candidate-delta-residual-bounded025-k8.pth",
    "yelp2018": "yelp2018-candidate-delta-residual-bounded025-k8.pth",
}


def default_ckpt(dataset_key: str) -> str:
    return os.path.join("checkpoints", CAP_CKPT[dataset_key])


def default_out(dataset_key: str) -> str:
    return os.path.join("checkpoints", LOW_CKPT[dataset_key])


def load_state(model: torch.nn.Module, path: str, device: torch.device) -> None:
    if not path or (not os.path.exists(path)):
        raise FileNotFoundError(f"ckpt not found: {path}")
    state = torch.load(path, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = (
        ".W_item_gate.weight",
        ".item_cross_scale",
        ".item_cos_scale",
        ".W_item_confidence.weight",
        ".item_confidence_cross_scale",
        ".item_confidence_cos_scale",
        ".W_item_suppress.weight",
        ".item_suppress_cross_scale",
        ".item_suppress_cos_scale",
        ".W_item_match.weight",
        ".W_social_match.weight",
        ".W_item_anchor.weight",
        ".item_anchor_cross_scale",
        ".item_anchor_cos_scale",
        ".W_interaction_adjust.weight",
        ".W_interaction_gate.weight",
        ".W_social_interaction.weight",
        ".W_conditional_gate.weight",
        ".conditional_cross_scale",
        ".conditional_cos_scale",
        ".dual_residual_mix_logit",
    )
    custom_allowed_missing = set()
    if hasattr(model, "warmstart_allowed_missing_keys"):
        custom_allowed_missing.update(
            model.warmstart_allowed_missing_keys()
        )
    invalid_missing = [
        key for key in incompatible.missing_keys
        if key not in custom_allowed_missing
        and not key.endswith(allowed_missing)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible checkpoint: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    if incompatible.missing_keys:
        print(
            "loaded checkpoint; missing item-aware residual parameters "
            "keep their mode-specific initialization",
            flush=True,
        )
    model.to(device)


def save_state(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)


def sample_unseen_negatives(
    batch_users: np.ndarray,
    train_item_sets: Dict[int, Set[int]],
    item_num: int,
    neg_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample K distinct negatives outside every training item of each user."""
    if neg_k < 1:
        raise ValueError("negative_k must be >= 1")

    negatives = rng.integers(
        0,
        item_num,
        size=(len(batch_users), neg_k),
        dtype=np.int64,
    )
    for row, uid in enumerate(batch_users):
        seen = train_item_sets.get(int(uid), set())
        if item_num - len(seen) < neg_k:
            raise ValueError(
                f"user {int(uid)} has fewer than {neg_k} unseen items"
            )
        chosen = set()
        for col in range(neg_k):
            while (
                int(negatives[row, col]) in seen
                or int(negatives[row, col]) in chosen
            ):
                negatives[row, col] = rng.integers(0, item_num)
            chosen.add(int(negatives[row, col]))
    return negatives


def train_one_run(data: GraphData, cfg: Dict[str, Any], device: torch.device, out_path: str) -> Tuple[float, Dict[str, float], int]:
    seed_everything(int(cfg["seed"]))

    model = ODSR(data, cfg, device, ablation_flags={}).to(device)
    finetune_role = str(cfg.get("finetune_role", "scratch"))
    init_ckpt = cfg.get("init_ckpt")
    freeze_backbone_epochs = int(cfg.get("freeze_backbone_epochs", 0))
    backbone_lr_scale = float(cfg.get("backbone_lr_scale", 0.1))
    if finetune_role not in {"scratch", "odsr_control", "item_aware"}:
        raise ValueError("unsupported finetune_role")
    if freeze_backbone_epochs < 0:
        raise ValueError("freeze_backbone_epochs must be >= 0")
    if backbone_lr_scale <= 0.0:
        raise ValueError("backbone_lr_scale must be > 0")
    if finetune_role != "scratch" and not init_ckpt:
        raise ValueError("finetune_role requires --init_ckpt")

    active_item_suffixes = (
        ".W_item_gate.weight",
        ".item_cross_scale",
        ".item_cos_scale",
        ".W_item_confidence.weight",
        ".item_confidence_cross_scale",
        ".item_confidence_cos_scale",
        ".W_item_suppress.weight",
        ".item_suppress_cross_scale",
        ".item_suppress_cos_scale",
    )
    custom_branch_names = None
    if hasattr(model, "warmstart_branch_parameter_names"):
        custom_branch_names = set(
            model.warmstart_branch_parameter_names()
        )
    base_params = []
    item_params = []
    if finetune_role == "scratch":
        opt = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    else:
        load_state(model, str(init_ckpt), device)
        # A neutral residual makes the warm-start model exactly reproduce the
        # official candidate-independent ODSR before any adaptation.
        with torch.no_grad():
            for layer in model.layers:
                layer.W_item_gate.weight.zero_()
                layer.item_cross_scale.zero_()
                layer.item_cos_scale.zero_()

        official_state = torch.load(str(init_ckpt), map_location="cpu")
        official_names = set(official_state.keys())
        for name, parameter in model.named_parameters():
            is_branch = (
                name in custom_branch_names
                if custom_branch_names is not None
                else name.endswith(active_item_suffixes)
            )
            if is_branch:
                parameter.requires_grad_(finetune_role == "item_aware")
                item_params.append(parameter)
            elif name in official_names:
                parameter.requires_grad_(freeze_backbone_epochs == 0)
                base_params.append(parameter)
            else:
                # Other historical ablation branches are absent from the
                # official checkpoint and inactive in the orth_gate runs.
                parameter.requires_grad_(False)

        parameter_groups = [{
            "params": base_params,
            "lr": float(cfg["lr"]) * backbone_lr_scale,
        }]
        if finetune_role == "item_aware":
            parameter_groups.append({
                "params": item_params,
                "lr": float(cfg["lr"]),
            })
        opt = optim.Adam(parameter_groups)

    users_arr = np.array([p[0] for p in data.train_data], dtype=np.int64)
    pos_arr = np.array([p[1] for p in data.train_data], dtype=np.int64)
    train_item_sets = {
        int(uid): set(items)
        for uid, items in data.train_items.items()
    }
    # Keep negative sampling independent from NumPy's epoch permutation and
    # PyTorch's initialization/dropout RNG streams.
    neg_rng = np.random.default_rng(int(cfg["seed"]) + 104729)
    candidate_pool_rng = np.random.default_rng(int(cfg["seed"]) + 209759)
    # In candidate-only mode this second generator consumes exactly the same
    # K=1 stream as an ODSR control.  Keeping it independent is essential:
    # taking column zero from a K=8 draw does not reproduce K=1 sampling.
    backbone_neg_rng = np.random.default_rng(int(cfg["seed"]) + 104729)

    bs = int(cfg["batch_size"])
    n_batch = max(1, len(users_arr) // bs)  # drop tail

    best_val = -1e18
    best_metrics: Dict[str, float] = {}
    best_ep = -1
    bad = 0

    max_epoch = int(cfg["max_epoch"])
    eval_interval = int(cfg["eval_interval"])
    patience = int(cfg["patience"])
    metric_key = str(cfg["metric"]).lower().strip()

    confidence_parameters = []
    if cfg["item_aware_mode"] == "orth_gate_confidence_direct":
        confidence_suffixes = (
            ".W_item_confidence.weight",
            ".item_confidence_cross_scale",
            ".item_confidence_cos_scale",
        )
        confidence_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.endswith(confidence_suffixes)
        ]

    if finetune_role != "scratch":
        initial_metrics = evaluate_full_ranking(
            model,
            data,
            device,
            k_list=(10, 20),
            eval_batch=int(cfg["eval_batch"]),
            eval_item_batch=int(cfg["eval_item_batch"]),
        )
        best_val = float(initial_metrics[metric_key])
        best_metrics = initial_metrics
        best_ep = 0
        save_state(model, out_path)
        print(
            f"eval init r20={initial_metrics['recall@20']:.6f} "
            f"n20={initial_metrics['ndcg@20']:.6f} "
            f"r10={initial_metrics['recall@10']:.6f} "
            f"n10={initial_metrics['ndcg@10']:.6f}",
            flush=True,
        )
        print(f"save {out_path} best={best_val:.6f} ep=0", flush=True)

    for ep in range(1, max_epoch + 1):
        if hasattr(model, "set_training_epoch"):
            model.set_training_epoch(ep)
        else:
            model.current_epoch = ep
        if confidence_parameters:
            confidence_active = (
                ep > int(cfg["confidence_warmup_epochs"])
            )
            model.confidence_warmup_active = not confidence_active
            for parameter in confidence_parameters:
                parameter.requires_grad_(confidence_active)
            if ep == 1 and not confidence_active:
                print(
                    "confidence phase=proposal_warmup "
                    f"frozen_epochs={cfg['confidence_warmup_epochs']} "
                    f"proposal_weight={cfg['proposal_warmup_loss_weight']}",
                    flush=True,
                )
            if ep == int(cfg["confidence_warmup_epochs"]) + 1:
                print(
                    "confidence phase=joint "
                    f"proposal_weight={cfg['proposal_aux_loss_weight']}",
                    flush=True,
                )
        if finetune_role != "scratch":
            # Reset epoch RNGs so the paired control and item-aware runs see
            # identical permutations, negatives, and dropout after warmup.
            seed_everything(int(cfg["seed"]) + ep)
            if ep == freeze_backbone_epochs + 1:
                for parameter in base_params:
                    parameter.requires_grad_(True)
                # Head-only warmup is a separate optimization phase.  Its
                # non-improving evaluations must not consume the patience
                # budget intended for joint backbone adaptation.
                bad = 0
                print(
                    f"unfreeze backbone epoch={ep} "
                    f"lr={float(cfg['lr']) * backbone_lr_scale:.8f}",
                    flush=True,
                )

        if (
            finetune_role == "odsr_control"
            and ep <= freeze_backbone_epochs
        ):
            print(
                f"epoch {ep}/{max_epoch} phase=neutral_warmup no_update",
                flush=True,
            )
            continue

        t0 = time.time()
        model.train()

        if finetune_role == "scratch":
            idx = np.random.permutation(len(users_arr))
            epoch_neg_rng = neg_rng
            epoch_candidate_pool_rng = candidate_pool_rng
            epoch_backbone_neg_rng = backbone_neg_rng
        else:
            epoch_rng = np.random.default_rng(int(cfg["seed"]) + ep)
            idx = epoch_rng.permutation(len(users_arr))
            epoch_neg_rng = np.random.default_rng(
                int(cfg["seed"]) + 104729 + ep
            )
            epoch_candidate_pool_rng = np.random.default_rng(
                int(cfg["seed"]) + 209759 + ep
            )
            epoch_backbone_neg_rng = np.random.default_rng(
                int(cfg["seed"]) + 104729 + ep
            )
        total = 0.0
        rec_component_totals: Dict[str, float] = {}

        for bi in range(n_batch):
            b = idx[bi * bs:(bi + 1) * bs]
            batch_users = users_arr[b]
            u = torch.LongTensor(batch_users).to(device)
            p = torch.LongTensor(pos_arr[b]).to(device)
            negative_k = int(cfg["negative_k"])
            sampled_candidate_k = negative_k
            if cfg["negative_strategy"] == "hard_topk_mean":
                sampled_candidate_k = int(cfg["hard_candidate_pool"])
            n = torch.from_numpy(sample_unseen_negatives(
                batch_users=batch_users,
                train_item_sets=train_item_sets,
                item_num=data.item_num,
                neg_k=sampled_candidate_k,
                rng=epoch_neg_rng,
            )).to(device)

            candidate_pool = None
            if getattr(model, "candidate_direct_history_expert", False):
                candidate_pool = torch.from_numpy(sample_unseen_negatives(
                    batch_users=batch_users,
                    train_item_sets=train_item_sets,
                    item_num=data.item_num,
                    neg_k=int(cfg["hard_pool_size"]),
                    rng=epoch_candidate_pool_rng,
                )).to(device)

            backbone_n = None
            if cfg["negative_gradient_mode"] == "candidate_only":
                backbone_n = torch.from_numpy(sample_unseen_negatives(
                    batch_users=batch_users,
                    train_item_sets=train_item_sets,
                    item_num=data.item_num,
                    neg_k=1,
                    rng=epoch_backbone_neg_rng,
                )).to(device)

            loss = model.compute_loss(
                u,
                p,
                n,
                backbone_neg=backbone_n,
                candidate_pool=candidate_pool,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            for name, value in model.last_rec_components.items():
                rec_component_totals[name] = (
                    rec_component_totals.get(name, 0.0)
                    + float(value.item())
                )

        avg = total / max(1, n_batch)
        component_text = ""
        if "backbone_k1" in rec_component_totals:
            component_text = (
                f" backbone_k1_bpr={rec_component_totals['backbone_k1'] / n_batch:.6f}"
                f" candidate_k{cfg['negative_k']}_bpr={rec_component_totals['candidate'] / n_batch:.6f}"
                f" candidate_weight={rec_component_totals['candidate_weight'] / n_batch:.3f}"
            )
            if "gate_saturation" in rec_component_totals:
                component_text += (
                    f" gate_saturation={rec_component_totals['gate_saturation'] / n_batch:.6f}"
                )
        elif "proposal" in rec_component_totals:
            component_text = (
                f" fused_bpr={rec_component_totals['fused'] / n_batch:.6f}"
                f" proposal_bpr={rec_component_totals['proposal'] / n_batch:.6f}"
                f" proposal_weight={rec_component_totals['proposal_weight'] / n_batch:.3f}"
            )
        elif "gate_change" in rec_component_totals:
            component_text = (
                f" fused_bpr={rec_component_totals['fused'] / n_batch:.6f}"
                f" gate_change={rec_component_totals['gate_change'] / n_batch:.6f}"
            )
        elif "gate_saturation" in rec_component_totals:
            component_text = (
                f" fused_bpr={rec_component_totals['fused'] / n_batch:.6f}"
                f" gate_saturation={rec_component_totals['gate_saturation'] / n_batch:.6f}"
            )
        elif "item" in rec_component_totals:
            component_text = (
                f" fused_bpr={rec_component_totals['fused'] / n_batch:.6f}"
                f" item_bpr={rec_component_totals['item'] / n_batch:.6f}"
                f" conditional_bpr={rec_component_totals['conditional'] / n_batch:.6f}"
            )
        print(
            f"epoch {ep}/{max_epoch} loss={avg:.6f}{component_text} "
            f"time={time.time() - t0:.1f}s",
            flush=True,
        )

        if ep % eval_interval != 0:
            continue

        metrics = evaluate_full_ranking(
            model,
            data,
            device,
            k_list=(10, 20),
            eval_batch=int(cfg["eval_batch"]),
            eval_item_batch=int(cfg["eval_item_batch"]),
        )
        val = float(metrics.get(metric_key, 0.0))
        print(
            f"eval Recall@10={metrics['recall@10']:.6f} "
            f"Recall@20={metrics['recall@20']:.6f} "
            f"NDCG@10={metrics['ndcg@10']:.6f} "
            f"NDCG@20={metrics['ndcg@20']:.6f} "
            f"monitor={metric_key}={val:.6f}",
            flush=True
        )

        if val > best_val:
            best_val = val
            best_metrics = metrics
            best_ep = ep
            bad = 0
            save_state(model, out_path)
            print(f"save {out_path} best={best_val:.6f} ep={best_ep}", flush=True)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop patience={patience}", flush=True)
                break

    return float(best_val), best_metrics, int(best_ep)


@torch.no_grad()
def eval_ckpt(data: GraphData, cfg: Dict[str, Any], device: torch.device, ckpt_path: str) -> Dict[str, float]:
    seed_everything(int(cfg["seed"]))
    model = ODSR(data, cfg, device, ablation_flags={}).to(device)
    load_state(model, ckpt_path, device)
    model.eval()
    return evaluate_full_ranking(
        model,
        data,
        device,
        k_list=(10, 20),
        eval_batch=int(cfg["eval_batch"]),
        eval_item_batch=int(cfg["eval_item_batch"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="douban-book", help="douban-book / epinions / yelp2018")
    ap.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="dataset parent directory (overrides SELFREC_DATA_ROOT)",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--social_norm", type=str, default="row", choices=["row", "sym"])

    ap.add_argument("--train", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint to load for eval (or resume manually)")
    ap.add_argument("--out", type=str, default=None, help="output checkpoint path for training")
    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument(
        "--finetune_role",
        type=str,
        default=None,
        choices=["scratch", "odsr_control", "item_aware"],
    )
    ap.add_argument("--freeze_backbone_epochs", type=int, default=None)
    ap.add_argument("--backbone_lr_scale", type=float, default=None)

    # optional overrides
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--reg", type=float, default=None)
    ap.add_argument("--ssl_reg", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--emb_size", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--max_epoch", type=int, default=None)
    ap.add_argument("--eval_interval", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--eval_batch", type=int, default=None)
    ap.add_argument("--eval_item_batch", type=int, default=None)
    ap.add_argument(
        "--item_aware_mode",
        type=str,
        default=None,
        choices=[
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
        ],
    )
    ap.add_argument("--negative_k", type=int, default=None)
    ap.add_argument(
        "--negative_strategy",
        type=str,
        default=None,
        choices=["random_mean", "hard_max", "hard_topk_mean"],
    )
    ap.add_argument(
        "--negative_gradient_mode",
        type=str,
        default=None,
        choices=["joint", "candidate_only"],
        help=(
            "joint: K negatives update the fused model; candidate_only: "
            "independent K=1 updates ODSR and K negatives update only the "
            "candidate readout"
        ),
    )
    ap.add_argument("--candidate_loss_weight", type=float, default=None)
    ap.add_argument("--hard_candidate_pool", type=int, default=None)
    ap.add_argument("--candidate_residual_bound", type=float, default=None)
    ap.add_argument("--candidate_recovery_scale", type=float, default=None)
    ap.add_argument("--candidate_confidence_bias", type=float, default=None)
    ap.add_argument("--signed_gate_scale", type=float, default=None)
    ap.add_argument("--gate_change_reg_weight", type=float, default=None)
    ap.add_argument("--gate_saturation_bound", type=float, default=None)
    ap.add_argument("--gate_saturation_reg_weight", type=float, default=None)
    ap.add_argument("--confidence_warmup_epochs", type=int, default=None)
    ap.add_argument("--proposal_warmup_loss_weight", type=float, default=None)
    ap.add_argument("--proposal_aux_loss_weight", type=float, default=None)
    ap.add_argument("--item_residual_bound", type=float, default=None)
    ap.add_argument("--conditional_residual_bound", type=float, default=None)
    ap.add_argument("--normalized_cross_scale", type=float, default=None)
    ap.add_argument("--item_cosine_groups", type=int, default=None)
    ap.add_argument("--item_branch_loss_weight", type=float, default=None)
    ap.add_argument("--conditional_branch_loss_weight", type=float, default=None)
    ap.add_argument(
        "--branch_aux_score_mode",
        type=str,
        default=None,
        choices=["full", "incremental"],
    )
    # Backward-compatible alias for completed/active K4 launch scripts.
    ap.add_argument("--hard_neg_k", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--metric", type=str, default=None)

    args = ap.parse_args()
    ds = args.dataset.lower().strip()
    if ds not in DATASET_CFG:
        raise ValueError(f"supported: {list(DATASET_CFG.keys())}")

    cfg = dict(COMMON)
    cfg.update(DATASET_CFG[ds])
    cfg["social_norm"] = args.social_norm

    for k in [
        "seed", "lr", "reg", "ssl_reg", "eps", "dropout",
        "batch_size", "n_layer", "emb_size", "tau",
        "max_epoch", "eval_interval", "patience", "eval_batch", "eval_item_batch",
        "item_aware_mode",
        "negative_k", "negative_strategy", "negative_gradient_mode",
        "candidate_loss_weight", "hard_candidate_pool",
        "candidate_residual_bound", "candidate_recovery_scale",
        "candidate_confidence_bias", "metric",
        "signed_gate_scale", "gate_change_reg_weight",
        "gate_saturation_bound", "gate_saturation_reg_weight",
        "confidence_warmup_epochs", "proposal_warmup_loss_weight",
        "proposal_aux_loss_weight",
        "item_residual_bound", "conditional_residual_bound",
        "normalized_cross_scale",
        "item_cosine_groups",
        "item_branch_loss_weight", "conditional_branch_loss_weight",
        "branch_aux_score_mode",
        "init_ckpt", "finetune_role", "freeze_backbone_epochs",
        "backbone_lr_scale",
    ]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    if args.hard_neg_k is not None:
        cfg["negative_k"] = args.hard_neg_k
        if args.negative_strategy is None:
            cfg["negative_strategy"] = "hard_max"

    if (
        cfg["negative_strategy"] == "hard_topk_mean"
        and int(cfg["hard_candidate_pool"]) < int(cfg["negative_k"])
    ):
        raise ValueError("hard_candidate_pool must be >= negative_k")

    device = torch.device(args.device)

    ckpt_path = args.ckpt if args.ckpt else default_ckpt(ds)
    out_path = args.out if args.out else default_out(ds)

    mode = "train" if args.train else "eval"
    print("mode:", mode, flush=True)
    print("dataset:", args.dataset, flush=True)
    print("device:", args.device, flush=True)
    print("social_norm:", cfg["social_norm"], flush=True)
    if cfg["item_aware_mode"] == "orth_gate_layer2_only":
        print("item_aware_layers: 2 only", flush=True)
    else:
        print("item_aware_layers: all", flush=True)
    print("item_aware_mode:", cfg["item_aware_mode"], flush=True)
    print("negative_k:", cfg["negative_k"], flush=True)
    print("negative_strategy:", cfg["negative_strategy"], flush=True)
    print("negative_gradient_mode:", cfg["negative_gradient_mode"], flush=True)
    if cfg["negative_gradient_mode"] == "candidate_only":
        print("backbone_negative_k: 1", flush=True)
        print("candidate_loss_weight:", cfg["candidate_loss_weight"], flush=True)
    print("finetune_role:", cfg["finetune_role"], flush=True)
    if cfg["finetune_role"] != "scratch":
        print("init_ckpt:", cfg["init_ckpt"], flush=True)
        print(
            "freeze_backbone_epochs:",
            cfg["freeze_backbone_epochs"],
            flush=True,
        )
        print("backbone_lr_scale:", cfg["backbone_lr_scale"], flush=True)
    if cfg["item_aware_mode"] in {
        "orth_gate_bounded_normalized_cross",
        "orth_gate_projected_bounded_cross",
    }:
        print(
            "normalized_cross_scale:",
            cfg["normalized_cross_scale"],
            flush=True,
        )
    if cfg["item_aware_mode"] == "orth_gate_group_cosine":
        print("item_cosine_groups:", cfg["item_cosine_groups"], flush=True)
    if cfg["item_aware_mode"] in {
        "orth_gate_dropped_recovery",
        "orth_gate_bidirectional_correction",
    }:
        print(
            "candidate_recovery_scale:",
            cfg["candidate_recovery_scale"],
            flush=True,
        )
    if cfg["item_aware_mode"] == "orth_gate_confidence_direct":
        print(
            "candidate_confidence_bias:",
            cfg["candidate_confidence_bias"],
            flush=True,
        )
        print(
            "confidence_warmup_epochs:",
            cfg["confidence_warmup_epochs"],
            flush=True,
        )
        print(
            "proposal_warmup_loss_weight:",
            cfg["proposal_warmup_loss_weight"],
            flush=True,
        )
        print(
            "proposal_aux_loss_weight:",
            cfg["proposal_aux_loss_weight"],
            flush=True,
        )
    if cfg["item_aware_mode"] == "orth_gate_signed_direct":
        print("signed_gate_scale:", cfg["signed_gate_scale"], flush=True)
        print(
            "gate_change_reg_weight:",
            cfg["gate_change_reg_weight"],
            flush=True,
        )
    if cfg["gate_saturation_reg_weight"] > 0.0:
        print(
            "gate_saturation_bound:",
            cfg["gate_saturation_bound"],
            flush=True,
        )
        print(
            "gate_saturation_reg_weight:",
            cfg["gate_saturation_reg_weight"],
            flush=True,
        )
    if cfg["item_aware_mode"] == "candidate_dual_separate_residual":
        print("item_residual_bound:", cfg["item_residual_bound"], flush=True)
        print(
            "conditional_residual_bound:",
            cfg["conditional_residual_bound"],
            flush=True,
        )
        print(
            "total_residual_bound:",
            cfg["item_residual_bound"] + cfg["conditional_residual_bound"],
            flush=True,
        )
        print(
            "item_branch_loss_weight:",
            cfg["item_branch_loss_weight"],
            flush=True,
        )
        print(
            "conditional_branch_loss_weight:",
            cfg["conditional_branch_loss_weight"],
            flush=True,
        )
        print("branch_aux_score_mode:", cfg["branch_aux_score_mode"], flush=True)
    else:
        print(
            "candidate_residual_bound:",
            cfg["candidate_residual_bound"],
            flush=True,
        )
    if cfg["negative_strategy"] == "hard_topk_mean":
        print("hard_candidate_pool:", cfg["hard_candidate_pool"], flush=True)
    print("negative_filter: all_train_history", flush=True)
    if args.train:
        print("out:", out_path, flush=True)
    else:
        print("ckpt:", ckpt_path, flush=True)

    if args.data_root:
        os.environ["SELFREC_DATA_ROOT"] = os.path.abspath(args.data_root)

    data = GraphData(args.dataset, device, social_norm=cfg["social_norm"], verbose=False)

    if args.train:
        best_val, best_metrics, best_ep = train_one_run(data, cfg, device, out_path)
        print("done", flush=True)
        print("best_epoch:", best_ep, flush=True)
        if best_metrics:
            print(
                f"best Recall@10={best_metrics['recall@10']:.6f} "
                f"Recall@20={best_metrics['recall@20']:.6f} "
                f"NDCG@10={best_metrics['ndcg@10']:.6f} "
                f"NDCG@20={best_metrics['ndcg@20']:.6f}",
                flush=True
            )
        print(f"best_{str(cfg['metric']).lower().strip()}={best_val:.6f}", flush=True)
    else:
        m = eval_ckpt(data, cfg, device, ckpt_path)
        print("eval", flush=True)
        print(
            f"Recall@10={m['recall@10']:.6f} Recall@20={m['recall@20']:.6f} "
            f"NDCG@10={m['ndcg@10']:.6f} NDCG@20={m['ndcg@20']:.6f}",
            flush=True
        )


if __name__ == "__main__":
    main()
