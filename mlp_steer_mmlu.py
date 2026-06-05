#!/usr/bin/env python3
"""
mlp_steer_mmlu.py

Use a trained MLP probe to steer MCQA decisions.

This version uses MLP (neural network) probes trained with mode == 'residual_prob'.
The steering logic is identical to probe_linear_steer_mmlu.py but uses MLP predictions.

Steering rule (residual-correction):
  Let p be the base model's per-option probabilities (length-4).
  Let r̂ be the MLP's predicted residuals for the 4 options of a question.
  We form a *convex* correction toward the ideal one-hot by:
      p'  =  Normalize(  p  +  γ * r̃  )
  where r̃ is r̂ centered to enforce ∑j r̃_j = 0 (stability), and γ ∈ [0,1]
  controls the strength of steering.

Outputs:
- preds_baseline.jsonl : baseline probs/pred/conf per question
- preds_steered.jsonl  : steered  probs/pred/conf (+ r_hat for debugging)
- metrics.json         : Accuracy, ECE, Brier, NLL for baseline vs steered
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from mlp_config import MLP_CONFIG

# Import prompt builders from oracle script (they handle all datasets correctly)
from mlp_steer_mmlu_oracle import (
    build_arc_fewshot_prompt, build_arc_fewshot_prompt_lmeval,
    build_hellaswag_fewshot_prompt, build_hellaswag_fewshot_prompt_lmeval,
    build_siqa_fewshot_prompt, build_siqa_fewshot_prompt_lmeval,
    build_openbookqa_fewshot_prompt, build_openbookqa_fewshot_prompt_lmeval,
    build_gsm_mc_fewshot_prompt, build_gsm_mc_fewshot_prompt_lmeval,
    build_math_mc_fewshot_prompt, build_math_mc_fewshot_prompt_lmeval,
    build_commonsenseqa_fewshot_prompt,
    build_sciq_fewshot_prompt_lmeval,
    build_race_fewshot_prompt_lmeval,
    get_generic_fewshot_examples,
)


# ----------------------------- feature augmentation -----------------------------

def augment_within_q_features(X: np.ndarray, ids: np.ndarray, center_within_q: bool, add_compete: bool) -> np.ndarray:
    """Apply within-question feature transformations."""
    if not (center_within_q or add_compete):
        return X
    
    X_out = X.copy()
    unique_qids = []
    for qid_full in ids:
        qid = qid_full.rsplit("_opt", 1)[0]
        if qid not in unique_qids:
            unique_qids.append(qid)
    
    if center_within_q:
        for qid in unique_qids:
            mask = np.array([qid_full.rsplit("_opt", 1)[0] == qid for qid_full in ids])
            if mask.sum() > 0:
                X_out[mask] = X_out[mask] - X_out[mask].mean(axis=0, keepdims=True)
    
    if add_compete:
        compete_feats = np.zeros_like(X_out)
        for qid in unique_qids:
            mask = np.array([qid_full.rsplit("_opt", 1)[0] == qid for qid_full in ids])
            indices = np.where(mask)[0]
            for i in indices:
                others = X_out[mask]
                others_mean = (others.sum(axis=0) - X_out[i]) / max(1, mask.sum() - 1)
                compete_feats[i] = X_out[i] - others_mean
        X_out = np.concatenate([X_out, compete_feats], axis=1)
    
    return X_out


def build_rank_feats(p_hat: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Build rank features (1-4) based on base model probabilities."""
    ranks = np.zeros(len(p_hat), dtype=np.float32)
    unique_qids = []
    for qid_full in ids:
        qid = qid_full.rsplit("_opt", 1)[0]
        if qid not in unique_qids:
            unique_qids.append(qid)
    
    for qid in unique_qids:
        mask = np.array([qid_full.rsplit("_opt", 1)[0] == qid for qid_full in ids])
        indices = np.where(mask)[0]
        p_vals = p_hat[indices]
        rank_order = np.argsort(-p_vals)
        for r, idx in enumerate(rank_order):
            ranks[indices[idx]] = r + 1
    
    return ranks.reshape(-1, 1)


def build_rank_onehot_feats(probs: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Build one-hot rank features."""
    ranks = build_rank_feats(probs, group_ids).flatten()
    onehot = np.zeros((len(ranks), 3), dtype=np.float32)
    for i, r in enumerate(ranks):
        if r == 2:
            onehot[i, 0] = 1
        elif r == 3:
            onehot[i, 1] = 1
        elif r == 4:
            onehot[i, 2] = 1
    return onehot


# ----------------------------- metrics -----------------------------

def ece_binary(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc  = y[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def brier_binary(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))

def nll_binary(probs: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def compute_classwise_ece(probs_list: List[np.ndarray], gold_list: List[int], num_bins: int = 25) -> dict:
    """
    Compute Class-wise Expected Calibration Error.
    
    Args:
        probs_list: List of probability distributions (variable length arrays)
        gold_list: List of gold labels
        num_bins: Number of bins
    
    Returns:
        Dictionary with per-class ECE and overall cwECE
    """
    gold = np.array(gold_list)
    # Find max number of classes across all examples
    num_classes = max(len(p) for p in probs_list)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    class_eces = []
    
    for c in range(num_classes):
        # Collect probabilities for class c (only from examples that have this class)
        class_probs = []
        class_correctness = []
        for i, probs in enumerate(probs_list):
            if c < len(probs):  # This example has class c
                class_probs.append(probs[c])
                class_correctness.append(1.0 if gold[i] == c else 0.0)
        
        if len(class_probs) == 0:
            class_eces.append(0.0)
            continue
        
        class_probs = np.array(class_probs)
        class_correctness = np.array(class_correctness)
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (class_probs > bin_lower) & (class_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = class_correctness[in_bin].mean()
                avg_confidence_in_bin = class_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        class_eces.append(float(ece))
    
    # cwECE is the average of per-class ECEs
    cwece = float(np.mean(class_eces))
    
    return {
        "cwECE": cwece,
        "Per_class_ECE": class_eces
    }


# ----------------------------- feature helpers -----------------------------

def zscore_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    return (X - mu) / sigma

def build_groups_from_ids(ids: List[str]) -> np.ndarray:
    return np.array([s.rsplit("_opt", 1)[0] for s in ids], dtype=object)

def augment_within_q_features(
    X: np.ndarray, ids_like: np.ndarray, center_within_q: bool, add_compete: bool
) -> np.ndarray:
    if not (center_within_q or add_compete):
        return X
    from collections import defaultdict
    qids = build_groups_from_ids(ids_like)
    buckets = defaultdict(list)
    for i, q in enumerate(qids):
        buckets[q].append(i)

    X_base = X.copy()
    if center_within_q:
        for _, idxs in buckets.items():
            I = np.asarray(idxs, dtype=int)
            X_base[I] -= X_base[I].mean(axis=0, keepdims=True)

    if not add_compete:
        return X_base

    X_comp = np.zeros_like(X_base, dtype=X_base.dtype)
    for _, idxs in buckets.items():
        I = np.asarray(idxs, dtype=int)
        S = X[I].sum(axis=0, keepdims=True)
        m = len(I)
        for i in I:
            avg_others = (S - X[i]) / max(m - 1, 1)
            X_comp[i] = X[i] - avg_others

    return np.concatenate([X_base, X_comp], axis=1)

def build_rank_feats(probs_4: np.ndarray, ids_like: np.ndarray) -> np.ndarray:
    from collections import defaultdict
    qids = build_groups_from_ids(ids_like)
    buckets = defaultdict(list)
    for i, q in enumerate(qids):
        buckets[q].append(i)
    out = np.zeros((len(ids_like), 1), dtype=np.float32)
    for _, idxs in buckets.items():
        I = np.asarray(idxs, dtype=int)
        p = probs_4[I]
        ranks = (-p).argsort().argsort() + 1
        out[I, 0] = ranks.astype(np.float32)
    return out

def build_rank_onehot_feats(probs_4: np.ndarray, ids_like: np.ndarray, max_choices: int = 4) -> np.ndarray:
    """Build one-hot rank features with configurable max_choices.
    
    Returns (N, max_choices-1) one-hot encoding for ranks 2..max_choices.
    Rank 1 is the implicit baseline (all zeros).
    """
    from collections import defaultdict
    qids = build_groups_from_ids(ids_like)
    buckets = defaultdict(list)
    for i, q in enumerate(qids):
        buckets[q].append(i)
    # Columns for rank 2, 3, ..., max_choices (rank 1 is implicit baseline)
    out = np.zeros((len(ids_like), max_choices - 1), dtype=np.float32)
    for _, idxs in buckets.items():
        I = np.asarray(idxs, dtype=int)
        p = probs_4[I]
        ranks = (-p).argsort().argsort() + 1
        for j, irow in enumerate(I):
            r = int(ranks[j])
            if 2 <= r <= max_choices:
                out[irow, r - 2] = 1.0  # rank 2 -> col 0, rank 3 -> col 1, etc.
    return out

def build_entropy_feats(probs_4: np.ndarray, ids_like: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    from collections import defaultdict
    qids = build_groups_from_ids(ids_like)
    buckets = defaultdict(list)
    for i, q in enumerate(qids):
        buckets[q].append(i)
    out = np.zeros((len(ids_like), 1), dtype=np.float32)
    for _, idxs in buckets.items():
        I = np.asarray(idxs, dtype=int)
        p = np.clip(probs_4[I].astype(np.float64), eps, 1.0)
        H = float(-(p * np.log(p)).sum())
        out[I, 0] = H
    return out

def build_margingap_feats(probs_4: np.ndarray, ids_like: np.ndarray, mode: str = "logit+top2", eps: float = 1e-12) -> np.ndarray:
    from collections import defaultdict
    qids = build_groups_from_ids(ids_like)
    buckets = defaultdict(list)
    for i, q in enumerate(qids):
        buckets[q].append(i)

    logp = np.log(np.clip(probs_4.astype(np.float64), eps, 1.0 - eps))
    cols = {"logit": 0, "logit+top2": 1, "logit+prob+top2": 2}[mode] + 1
    C = 1 + cols
    feats = np.zeros((len(probs_4), C), dtype=np.float32)

    for _, idxs in buckets.items():
        I = np.asarray(idxs, dtype=int)
        p_q, lp_q = probs_4[I], logp[I]
        top2 = np.partition(p_q, -2)[-2:]
        top2_gap = float(top2[-1] - top2[-2]) if len(top2) >= 2 else 0.0

        max_lp = np.array([np.max(np.delete(lp_q, j)) for j in range(len(I))], dtype=np.float64)
        logit_margin = lp_q - max_lp
        feats[I, 0] = logit_margin.astype(np.float32)

        c = 1
        if "prob" in mode:
            max_p = np.array([np.max(np.delete(p_q, j)) for j in range(len(I))], dtype=np.float64)
            prob_gap = p_q - max_p
            feats[I, c] = prob_gap.astype(np.float32); c += 1
        if "top2" in mode:
            feats[I, c] = top2_gap
    return feats


def build_logit_features_single_question(
    probs: np.ndarray,
    *,
    temperature_scales: List[float],
    use_logp: bool,
    use_is_pred: bool,
    use_rank: bool,
    use_temp_scaled: bool,
    use_pmax: bool,
    use_margin: bool,
    use_entropy: bool,
    use_centered_p: bool,
    use_compete: bool,
    use_option_onehot: bool,
    max_choices: int = 4,
) -> np.ndarray:
    """Baseline-style logit/probability-derived features for one question.

    Returns:
      (n_choices, D_extra) feature matrix appended to activation features.

    Note: Always includes p_j as the first feature when this block is enabled.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n_choices = len(probs)
    pred_j = int(np.argmax(probs))

    temps = [float(t) for t in temperature_scales]
    temps_extra = [t for t in temps if abs(t - 1.0) > 1e-8]

    feats: List[List[float]] = []
    for j in range(n_choices):
        p_j = float(probs[j])
        row: List[float] = []

        # Always include p_j
        row.append(p_j)

        if use_logp:
            row.append(float(np.log(max(p_j, 1e-12))))

        if use_is_pred:
            row.append(1.0 if j == pred_j else 0.0)

        if use_rank:
            row.append(float(1 + np.sum(probs > p_j) + 0.5 * np.sum(probs == p_j)))

        if use_temp_scaled:
            for T in temps_extra:
                logits_approx = np.log(probs + 1e-12) / float(T)
                probs_T = np.exp(logits_approx - logits_approx.max())
                probs_T = probs_T / probs_T.sum()
                row.append(float(probs_T[j]))

        if use_pmax:
            row.append(float(np.max(probs)))

        if use_margin:
            sorted_probs = np.sort(probs)[::-1]
            row.append(float(sorted_probs[0] - sorted_probs[1]))

        if use_entropy:
            row.append(float(-np.sum(probs * np.log(probs + 1e-12))))

        if use_centered_p:
            row.append(float(p_j - float(np.mean(probs))))

        if use_compete:
            avg_others = float((np.sum(probs) - p_j) / max(n_choices - 1, 1))
            row.append(float(p_j - avg_others))

        if use_option_onehot:
            onehot = [0.0] * max_choices
            if j < max_choices:
                onehot[j] = 1.0
            row.extend(onehot)

        feats.append(row)

    return np.asarray(feats, dtype=np.float32)


# ----------------------------- LM scoring -----------------------------

@torch.no_grad()
def option_logprob_and_hidden(
    model, tok, prompt_ids, answer_ids, layers, pool="answer_mean"
) -> Tuple[float, Dict[int, np.ndarray]]:
    """Compute log probability and hidden states for an answer option.
    
    Uses token-length normalization matching mlp_steer_mmlu_oracle.py.
    """
    device = model.device
    input_ids = torch.tensor(prompt_ids + answer_ids, device=device).unsqueeze(0)
    attn_mask = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)

    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    prompt_len = len(prompt_ids)
    start = prompt_len - 1
    end = labels.size(1)

    logprobs_all = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    ans_logprobs = logprobs_all[:, start:end]
    logprob_len_norm = ans_logprobs.sum(dim=1) / (end - start)
    logprob = float(logprob_len_norm.item())

    feats = {}
    for li in layers:
        h = out.hidden_states[li].squeeze(0)
        if pool == "answer_mean":
            pooled = h[prompt_len:, :].mean(dim=0)
        elif pool == "answer_final":
            pooled = h[prompt_len + len(answer_ids) - 1, :]
        else:
            raise ValueError("pool must be {'answer_mean','answer_final'}")
        feats[li] = pooled.detach().float().cpu().numpy()

    return logprob, feats

def softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z -= z.max()
    return np.exp(z) / np.exp(z).sum()

def _parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [float(x) for x in s.split(",")]


# ----------------------------- MLP probe model -----------------------------

class MLPProbe(nn.Module):
    """MLP architecture for prediction."""
    def __init__(self, config: Dict):
        super().__init__()
        layers = []
        input_dim = config["input_dim"]
        
        for hidden_dim in config["hidden_dims"]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if config.get("batch_norm", False):
                layers.append(nn.BatchNorm1d(hidden_dim))
            if config.get("layer_norm", False):
                layers.append(nn.LayerNorm(hidden_dim))
            
            if config["activation"] == "relu":
                layers.append(nn.ReLU())
            elif config["activation"] == "gelu":
                layers.append(nn.GELU())
            elif config["activation"] == "tanh":
                layers.append(nn.Tanh())
            
            if config.get("dropout", 0.0) > 0:
                layers.append(nn.Dropout(config["dropout"]))
            
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, config["output_dim"]))
        
        # Output activation (optional, for bounded outputs)
        self.output_activation = config.get("output_activation")
        self.output_scale = config.get("output_scale", 1.0)
        
        if self.output_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.network(x).squeeze(-1)
        
        # Apply output scaling if using bounded activation
        if self.output_activation in ["tanh", "sigmoid"]:
            output = output * self.output_scale
        
        return output


class MLPProbeModel:
    """Wrapper for MLP probe with normalization and inference."""
    
    def __init__(self, artifact_path: str, device: str = "cuda"):
        with open(artifact_path, "rb") as f:
            art = pickle.load(f)
        
        # Core required fields shared by single-layer and concat artifacts
        required = ["mu", "sigma", "model_state_dict", "config"]
        for k in required:
            if k not in art:
                raise KeyError(f"MLP artifact missing key '{k}'")
        
        self.mu = np.asarray(art["mu"]).astype(np.float32)
        self.sigma = np.asarray(art["sigma"]).astype(np.float32)
        self.config = art["config"]
        self.mode = art.get("mode", "residual_prob")
        self.device = device

        # Layer information: support either single-layer or multi-layer concat artifacts
        if "layer_indices" in art:
            self.layer_indices = list(art["layer_indices"])
        elif "layer_index" in art:
            self.layer_index = int(art["layer_index"])
            self.layer_indices = [self.layer_index]
        else:
            raise KeyError("MLP artifact must contain 'layer_index' or 'layer_indices'")

        # For backward compatibility, always expose layer_index (first layer)
        if not hasattr(self, "layer_index"):
            self.layer_index = int(self.layer_indices[0])
        
        # Load meta flags for feature augmentation
        self.meta = art.get("meta", {})
        self.use_center_within_q = self.meta.get("use_center_within_q", False)
        self.use_compete = self.meta.get("use_compete", False)
        self.use_rankfeat = self.meta.get("use_rankfeat", False)
        self.use_rank_onehot = self.meta.get("use_rank_onehot", False)

        # BUGFIX: Verify feature dimensions match actual probe input
        # If input_dim is 4097 (4096 + 1), then only rankfeat is used, not rank_onehot
        input_dim = self.config.get("input_dim", len(self.mu))
        if input_dim == 4097 and self.use_rankfeat and self.use_rank_onehot:
            # 4096 (hidden) + 1 (rankfeat) = 4097, so rank_onehot was NOT actually used
            self.use_rank_onehot = False

        # Optional baseline logit/prob feature block
        self.use_logit_block = self.meta.get("use_logit_block", False)
        self.logit_temperature_scales = self.meta.get("logit_temperature_scales", [0.7, 1.0, 1.3])
        self.logit_use_logp = self.meta.get("logit_use_logp", False)
        self.logit_use_is_pred = self.meta.get("logit_use_is_pred", False)
        self.logit_use_rank = self.meta.get("logit_use_rank", False)
        self.logit_use_temp_scaled = self.meta.get("logit_use_temp_scaled", False)
        self.logit_use_pmax = self.meta.get("logit_use_pmax", False)
        self.logit_use_margin = self.meta.get("logit_use_margin", False)
        self.logit_use_entropy = self.meta.get("logit_use_entropy", False)
        self.logit_use_centered_p = self.meta.get("logit_use_centered_p", False)
        self.logit_use_compete = self.meta.get("logit_use_compete", False)
        self.logit_use_option_onehot = self.meta.get("logit_use_option_onehot", False)
        
        # Infer max_choices from probe metadata or from expected input dimension
        # This is critical for construct data which may have 5 choices vs MMLU's 4
        self.max_choices = self.meta.get("max_choices", None)
        if self.max_choices is None:
            # Infer max_choices by computing what value would give the correct input_dim
            # This works for both single-layer and multi-layer probes
            input_dim = self.config.get("input_dim", len(self.mu))
            
            # Count fixed augmentation features (not dependent on max_choices)
            fixed_feats = 0
            if self.use_rankfeat:
                fixed_feats += 1
            
            if self.use_logit_block:
                fixed_feats += 1  # p_j is always included
                if self.logit_use_logp:
                    fixed_feats += 1
                if self.logit_use_is_pred:
                    fixed_feats += 1
                if self.logit_use_rank:
                    fixed_feats += 1
                if self.logit_use_temp_scaled:
                    temps = self.logit_temperature_scales
                    temps_extra = [t for t in temps if abs(t - 1.0) > 1e-8]
                    fixed_feats += len(temps_extra)
                if self.logit_use_pmax:
                    fixed_feats += 1
                if self.logit_use_margin:
                    fixed_feats += 1
                if self.logit_use_entropy:
                    fixed_feats += 1
                if self.logit_use_centered_p:
                    fixed_feats += 1
                if self.logit_use_compete:
                    fixed_feats += 1
            
            # Compute variable features per max_choices value
            # rank_onehot: max_choices - 1 columns
            # option_onehot: max_choices columns
            def compute_variable_feats(mc):
                v = 0
                if self.use_rank_onehot:
                    v += mc - 1
                if self.logit_use_option_onehot:
                    v += mc
                return v
            
            # Try max_choices from 2 to 8 and find which one matches
            # by checking if (input_dim - fixed_feats - variable_feats) gives a reasonable base_dim
            best_match = 4  # default
            for mc in range(2, 9):
                variable_feats = compute_variable_feats(mc)
                remainder = input_dim - fixed_feats - variable_feats
                
                # For single-layer probes, base_dim should be a known hidden size (4096 for 7B models)
                # or divisible by common hidden sizes
                # For multi-layer probes with layer_dims, we can verify exactly
                if "layer_dims" in art:
                    layer_dims = art["layer_dims"]
                    expected_base = sum(layer_dims)
                    if self.use_compete:
                        expected_base *= 2
                    if remainder == expected_base:
                        best_match = mc
                        break
                else:
                    # For single-layer: check if remainder is a common hidden size or matches config
                    # Common hidden sizes: 4096 (7B), 5120 (13B), etc.
                    if remainder > 0 and remainder % 128 == 0:  # Hidden sizes are typically multiples of 128
                        # Additional check: if use_compete, remainder should be 2x a common size
                        if self.use_compete:
                            if remainder % 2 == 0 and (remainder // 2) in [4096, 3584, 5120, 8192]:
                                best_match = mc
                                break
                        else:
                            if remainder in [4096, 3584, 5120, 8192]:
                                best_match = mc
                                break
            
            self.max_choices = best_match
        
        # Load MLP model
        self.model = MLPProbe(self.config).to(device)
        self.model.load_state_dict(art["model_state_dict"])
        self.model.eval()
    
    def _build_hidden_concat(self, feats_per_option: Dict[int, np.ndarray]) -> np.ndarray:
        """Extract and concatenate hidden states for all required layers."""
        xs = []
        for L in self.layer_indices:
            if L not in feats_per_option:
                raise KeyError(f"Missing layer {L} in collected feats.")
            xs.append(np.asarray(feats_per_option[L], dtype=np.float32))
        if len(xs) == 1:
            return xs[0]
        return np.concatenate(xs, axis=0)
    
    def predict_residuals(self, X_rows: np.ndarray, ids: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
        """
        Predict residuals for the 4 options using the MLP.
        
        Args:
            X_rows: Raw feature matrix (4, D_raw)
            ids: Question IDs for the 4 options (4,)
            p_hat: Base model probabilities for the 4 options (4,)
        
        Returns:
            Predicted residuals (4,)
        """
        # Apply feature augmentation (same as training)
        X_aug = augment_within_q_features(X_rows, ids, 
                                          center_within_q=self.use_center_within_q,
                                          add_compete=self.use_compete)
        
        # Add rank features if used during training
        if self.use_rankfeat:
            rankfeat = build_rank_feats(p_hat, ids)
            X_aug = np.concatenate([X_aug, rankfeat], axis=1)
        
        if self.use_rank_onehot:
            rank_onehot = build_rank_onehot_feats(p_hat, ids, max_choices=self.max_choices)
            X_aug = np.concatenate([X_aug, rank_onehot], axis=1)

        if self.use_logit_block:
            logit_feats = build_logit_features_single_question(
                p_hat,
                temperature_scales=self.logit_temperature_scales,
                use_logp=self.logit_use_logp,
                use_is_pred=self.logit_use_is_pred,
                use_rank=self.logit_use_rank,
                use_temp_scaled=self.logit_use_temp_scaled,
                use_pmax=self.logit_use_pmax,
                use_margin=self.logit_use_margin,
                use_entropy=self.logit_use_entropy,
                use_centered_p=self.logit_use_centered_p,
                use_compete=self.logit_use_compete,
                use_option_onehot=self.logit_use_option_onehot,
                max_choices=self.max_choices,
            )
            X_aug = np.concatenate([X_aug, logit_feats], axis=1)
        
        # Z-score normalization
        Z = (X_aug - self.mu) / self.sigma
        
        # MLP inference
        with torch.no_grad():
            X_tensor = torch.FloatTensor(Z).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Note: Output scaling is handled in model forward pass
        # If using tanh with output_scale=2.0, outputs are in [-2, 2]
        
        return np.asarray(predictions, dtype=np.float64)


# ----------------------------- steering combiners -----------------------------

def residual_correction_from_delta(p_base: np.ndarray, delta: np.ndarray, gamma: float) -> np.ndarray:
    """
    Same as residual_correction but expects an already-combined, centered delta
    (sum(delta)=0). Applies gamma directly (uncapped) and renormalizes.
    """
    p = np.asarray(p_base, dtype=np.float64)
    d = np.asarray(delta,   dtype=np.float64)
    
    # Apply gamma directly, no cap
    p_new = p + gamma * d
    
    # Clip and renormalize
    p_new = np.clip(p_new, 0.0, None)
    if p_new.sum() <= 0:
        return p
    return p_new / p_new.sum()

def residual_correction(p_base: np.ndarray, r_hat: np.ndarray, gamma: float) -> np.ndarray:
    """
    p' = Normalize( p + γ * r̃ ), with r̃ centered to sum to zero.
    Gamma is applied directly (uncapped).
    """
    p = np.asarray(p_base, dtype=np.float64)
    r = np.asarray(r_hat, dtype=np.float64)

    # Center residuals to enforce zero-sum (stability under imperfect probe)
    r_tilde = r - r.mean()

    # Apply gamma directly, no cap
    p_new = p + gamma * r_tilde

    # Clip and renormalize
    p_new = np.clip(p_new, 0.0, None)
    if p_new.sum() <= 0:
        # Fallback: if degenerate, return original p
        return p
    return p_new / p_new.sum()


def poe_combine(p_base: np.ndarray, s_prob: np.ndarray, alpha: float, strength: float) -> np.ndarray:
    """Product-of-Experts combiner (used if artifact is mode='correct')."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    lam = float(max(strength, 0.0))
    log_p = np.log(np.clip(p_base, 1e-9, 1.0)) * (1.0 - alpha) + np.log(np.clip(s_prob, 1e-9, 1.0)) * (alpha * lam)
    return softmax_np(log_p)


# ----------------------------- prompt formatting -----------------------------

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

def build_prompt_legacy(question: str, choices: List[str]) -> str:
    """Legacy format: Q: {question}\nA) {choice}\n...\nAnswer:"""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]  # Support up to 8 choices
    lines = [f"Q: {question}"]
    for L, c in zip(letters, choices):
        lines.append(f"{L}) {c}")
    lines.append("Answer:")
    return "\n".join(lines)

def build_prompt_lmeval(question: str, choices: List[str]) -> str:
    """LM-Eval harness format: {question}\nA. {choice}\n...\nAnswer:"""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]  # Support up to 8 choices
    lines = [question.strip()]
    for L, c in zip(letters, choices):
        lines.append(f"{L}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)

def build_fewshot_prompt(
    question: str,
    choices: List[str],
    fewshot_examples: List[Dict],
    prompt_format: str = "lmeval",
) -> str:
    """
    Build few-shot prompt following lm-eval harness format for MMLU.
    Examples are separated by double newlines.
    """
    build_fn = build_prompt_lmeval if prompt_format == "lmeval" else build_prompt_legacy
    prompt_parts = []
    
    # Add few-shot examples with answers
    for ex in fewshot_examples:
        ex_question = ex["question"]
        ex_choices = ex["choices"]
        ex_answer_idx = int(ex["answer"])
        correct_letter = LETTERS[ex_answer_idx]
        
        ex_prompt = build_fn(ex_question, ex_choices)
        # Replace "Answer:" with "Answer: {letter}"
        ex_prompt = ex_prompt[:-len("Answer:")] + f"Answer: {correct_letter}"
        prompt_parts.append(ex_prompt)
    
    # Add test question (without answer)
    test_prompt = build_fn(question, choices)
    prompt_parts.append(test_prompt)
    
    return "\n\n".join(prompt_parts)

def get_fewshot_examples(dataset_name: str, subject: str, num_fewshot: int) -> List[Dict]:
    """
    Get few-shot examples from the dev split following lm-eval's first_n sampler.
    """
    if num_fewshot == 0:
        return []
    
    # Load dev split for few-shot examples
    dev_ds = load_dataset(dataset_name, subject, split="dev")
    
    n_available = min(num_fewshot, len(dev_ds))
    examples = []
    for i in range(n_available):
        examples.append({
            "question": dev_ds[i]["question"],
            "choices": dev_ds[i]["choices"],
            "answer": dev_ds[i]["answer"],
        })
    
    return examples

def build_prompt(question: str, choices: List[str], prompt_format: str = "legacy",
                 fewshot_examples: List[Dict] = None) -> str:
    """Build prompt with specified format and optional few-shot examples."""
    if fewshot_examples:
        return build_fewshot_prompt(question, choices, fewshot_examples, prompt_format)
    elif prompt_format == "lmeval":
        return build_prompt_lmeval(question, choices)
    else:
        return build_prompt_legacy(question, choices)


# ----------------------------- main loop -----------------------------

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if args.dtype == "auto":
        dtype = "auto"
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    ).eval()
    print(f"Model loaded: {args.model_id} (dtype={model.dtype})")

    # Load MLP probe (single layer only)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    probe = MLPProbeModel(args.probe_pkl, device=device)
    print(f"[probe] Loaded MLP probe from {args.probe_pkl}")
    print(f"  - Layer: {probe.layer_index}")
    print(f"  - Config: {probe.config['hidden_dims']}")
    print(f"  - Mode: {probe.mode}")

    # Get layers needed for this probe
    union_layers = probe.layer_indices
    print(f"[layers] Using layers: {union_layers}")

    # Load dataset (optionally with subset/config)
    if args.subset and args.subset.lower() not in ["none", "default", ""]:
        ds = load_dataset(args.dataset, args.subset, split=args.split, trust_remote_code=True)
    else:
        ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    
    # Detect dataset format
    is_mmlu = "cais/mmlu" in args.dataset or "mmlu" in args.dataset.lower()
    is_arc = "ai2_arc" in args.dataset or "arc" in args.dataset.lower()
    is_medmcqa = "medmcqa" in args.dataset.lower()
    is_hellaswag = "hellaswag" in args.dataset.lower()
    is_piqa = "piqa" in args.dataset.lower()
    is_siqa = "social_i_qa" in args.dataset.lower() or "siqa" in args.dataset.lower()
    is_commonsenseqa = "commonsense_qa" in args.dataset.lower()
    is_gsm_mc = "gsm8k-mc" in args.dataset.lower() or "gsm-mc" in args.dataset.lower()
    is_math_mc = "math-mc" in args.dataset.lower() and "gsm" not in args.dataset.lower()
    is_openbookqa = "openbookqa" in args.dataset.lower()
    is_sciq = "sciq" in args.dataset.lower()
    is_race = "race" in args.dataset.lower()
    
    # Filter to test split if requested (60/20/20 split)
    original_indices = None  # Track original dataset indices for question IDs
    if args.split_ids_dir:
        test_indices_path = os.path.join(args.split_ids_dir, "test_row_indices.npy")
        if not os.path.exists(test_indices_path):
            raise FileNotFoundError(f"Test row indices not found at {test_indices_path}. "
                                    f"Run create_train_val_test_split.py first.")
        
        # Load row-level indices (these index into probe_data.npz rows, not dataset questions)
        # probe_data.npz has 4 rows per question (one per option)
        subset_row_indices = np.load(test_indices_path)
        n_rows = len(subset_row_indices)
        n_questions = n_rows // 4  # 4 options per question
        
        # Each question has 4 consecutive rows, so question_idx = row_idx // 4
        # Take every 4th row index and divide by 4 to get question indices
        question_indices = (subset_row_indices[::4] // 4).tolist()
        original_indices = question_indices  # Keep track of original indices
        ds = ds.select(question_indices)
        print(f"[split] Using test split: {n_questions} questions ({n_rows} rows) from {args.split_ids_dir}")
    else:
        print(f"[split] No split specified - using ALL {len(ds)} questions")
        original_indices = list(range(len(ds)))  # All indices
    
    if args.max_examples and args.max_examples < len(ds):
        ds = ds.select(range(args.max_examples))
        if original_indices:
            original_indices = original_indices[:args.max_examples]

    letters = ["A","B","C","D","E","F","G","H"]
    
    # Cache for few-shot examples (per subject for MMLU)
    fewshot_cache = {}
    fewshot_examples = None
    
    # Load few-shot examples for non-MMLU benchmarks (before main loop)
    if args.num_fewshot > 0 and not is_mmlu:
        print(f"[few-shot] Loading {args.num_fewshot} examples from train split...")
        if is_arc:
            if args.subset and args.subset.lower() not in ["none", "default", ""]:
                train_ds = load_dataset(args.dataset, args.subset, split="train", trust_remote_code=True)
            else:
                train_ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        elif is_hellaswag:
            train_ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        elif is_siqa:
            train_ds = load_dataset("social_i_qa", split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        elif is_gsm_mc:
            train_ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        elif is_math_mc:
            train_ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        elif is_openbookqa:
            if args.subset and args.subset.lower() not in ["none", "default", ""]:
                train_ds = load_dataset(args.dataset, args.subset, split="train", trust_remote_code=True)
            else:
                train_ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        elif is_sciq:
            train_ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        elif is_race:
            if args.subset and args.subset.lower() not in ["none", "default", ""]:
                train_ds = load_dataset(args.dataset, args.subset, split="train", trust_remote_code=True)
            else:
                train_ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        elif is_commonsenseqa:
            train_ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
            fewshot_examples = get_generic_fewshot_examples(train_ds, args.num_fewshot)
        if fewshot_examples:
            print(f"[few-shot] Loaded {len(fewshot_examples)} examples")
    
    # Print prompt format info
    print(f"[prompt] format={args.prompt_format}, scoring_mode={args.scoring_mode}, num_fewshot={args.num_fewshot}")

    f_base  = open(out_dir / "preds_baseline.jsonl", "w")
    f_steer = open(out_dir / "preds_steered.jsonl",  "w")

    acc_base, acc_steer = [], []
    conf_base, conf_steer = [], []
    probs_base, probs_steer = [], []  # For cwECE calculation
    gold_labels = []  # For cwECE calculation

    for idx, ex in enumerate(ds):
        # Handle different dataset formats
        if is_mmlu:
            q = ex["question"]
            choices = ex["choices"]
            gold_idx = int(ex["answer"])  # MMLU uses integer index directly
            gold_letter = letters[gold_idx]
            subject = ex.get("subject", args.subset)  # Get actual subject
        elif is_arc:
            # ARC format (allenai/ai2_arc):
            #   question: str
            #   choices: {"text": [...], "label": [...]} where label matches answerKey
            q = ex["question"]
            choices = ex["choices"]["text"]
            choice_labels = ex["choices"].get("label", ["A","B","C","D"])
            gold_letter = ex["answerKey"]
            # answerKey is one of the labels (e.g., 'A','B','C','D' or '1','2','3','4')
            if gold_letter in choice_labels:
                gold_idx = choice_labels.index(gold_letter)
            else:
                # Fallback: try mapping via letters if labels are standard A-D
                gold_idx = letters.index(gold_letter)
            subject = args.subset
        elif is_medmcqa:
            # MedMCQA format (openlifescienceai/medmcqa):
            #   question: str
            #   opa/opb/opc/opd: option texts
            #   cop: correct option index (1-4)
            q = ex["question"]
            choices = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
            gold_idx = int(ex["cop"]) - 1
            subject = ex.get("subject_name", args.subset)
        elif is_hellaswag:
            # HellaSwag format (Rowan/hellaswag):
            #   context fields (ctx/ctx_a/ctx_b) plus 4 endings and label 0-3
            if "ctx_a" in ex and "ctx_b" in ex:
                q = f"{ex['ctx_a']} {ex['ctx_b']}".strip()
            elif "ctx" in ex:
                q = ex["ctx"]
            else:
                q = ex.get("context", "")
            choices = list(ex["endings"])
            gold_idx = int(ex["label"])
            gold_letter = letters[gold_idx]
            subject = args.subset
        elif is_piqa:
            # PIQA format: goal, sol1, sol2, label (0 or 1)
            q = ex["goal"]
            choices = [ex["sol1"], ex["sol2"]]
            gold_idx = int(ex["label"])
            subject = "piqa"
        elif is_siqa:
            # Social IQA format: context, question, answerA/B/C, label (1/2/3 as string)
            q = f"{ex['context']} {ex['question']}"
            choices = [ex["answerA"], ex["answerB"], ex["answerC"]]
            gold_idx = int(ex["label"]) - 1  # label is 1-indexed string
            subject = "siqa"
        elif is_commonsenseqa:
            # CommonsenseQA format: question, choices dict with text/label, answerKey
            q = ex["question"]
            choices = ex["choices"]["text"]
            choice_labels = ex["choices"]["label"]
            gold_letter = ex["answerKey"]
            gold_idx = choice_labels.index(gold_letter)
            subject = "commonsenseqa"
        elif is_gsm_mc:
            # GSM-MC format: Question, A/B/C/D choices, Answer letter
            q = ex["Question"]
            choices = [ex["A"], ex["B"], ex["C"], ex["D"]]
            gold_letter = ex["Answer"]
            gold_idx = letters.index(gold_letter)
            subject = "gsm_mc"
        elif is_math_mc:
            # MATH-MC format: Question, A/B/C/D choices, Answer letter, Level, Type
            q = ex["Question"]
            choices = [ex["A"], ex["B"], ex["C"], ex["D"]]
            gold_letter = ex["Answer"]
            gold_idx = letters.index(gold_letter)
            subject = ex.get("Type", "math_mc")
        elif is_sciq:
            # SciQ format (allenai/sciq):
            #   question: str
            #   distractor1/2/3: wrong answers
            #   correct_answer: correct answer text
            #   support: context (optional)
            q = ex["question"]
            sciq_support = ex.get("support", "").strip()  # Store for lm-eval prompt
            choices = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
            gold_idx = 0  # correct_answer is always first in our list
            gold_letter = letters[gold_idx]
            subject = "sciq"
        elif is_race:
            # RACE format (ehovy/race):
            #   article: context passage
            #   question: question text
            #   options: list of 4 answer options
            #   answer: letter A/B/C/D
            race_article = ex["article"]  # Store for lm-eval prompt
            q = ex["question"]  # Just the question, article handled in prompt builder
            choices = ex["options"]
            gold_letter = ex["answer"]
            gold_idx = letters.index(gold_letter)
            subject = "race"
        elif is_openbookqa:
            # OpenBookQA format
            q = ex["question_stem"]
            choices = ex["choices"]["text"]
            gold_letter = ex["answerKey"]
            gold_idx = letters.index(gold_letter)
            subject = args.subset
        else:
            # Generic fallback: try common field names
            q = ex.get("question", ex.get("question_stem", ""))
            if "choices" in ex:
                if isinstance(ex["choices"], dict) and "text" in ex["choices"]:
                    choices = ex["choices"]["text"]
                else:
                    choices = ex["choices"]
            else:
                choices = [ex.get(k, "") for k in ["A", "B", "C", "D"] if k in ex]
            gold_letter = ex.get("answerKey", ex.get("answer", "A"))
            if gold_letter in letters:
                gold_idx = letters.index(gold_letter)
            else:
                gold_idx = int(gold_letter) if gold_letter.isdigit() else 0
            subject = args.subset

        # Get few-shot examples if needed (MMLU uses per-subject caching)
        # Prompt format depends on scoring_mode:
        # - "lmeval": use lm-eval-harness style prompts (answer text scoring)
        # - "letter": use MMLU-style prompts (show options, letter scoring)
        use_letter_scoring = (args.scoring_mode == "letter")
        
        if args.num_fewshot > 0 and is_mmlu:
            # MMLU always uses letter scoring with options shown
            if subject not in fewshot_cache:
                fewshot_cache[subject] = get_fewshot_examples(args.dataset, subject, args.num_fewshot)
            mmlu_fewshot = fewshot_cache[subject]
            prompt = build_prompt(q, choices, args.prompt_format, mmlu_fewshot)
        elif fewshot_examples is not None:
            # Use benchmark-specific prompt builders based on scoring mode
            if use_letter_scoring:
                # MMLU-style: show options, answer with letter
                if is_arc:
                    prompt = build_arc_fewshot_prompt(q, choices, fewshot_examples)
                elif is_hellaswag:
                    prompt = build_hellaswag_fewshot_prompt(q, choices, fewshot_examples)
                elif is_siqa:
                    prompt = build_siqa_fewshot_prompt(q, choices, fewshot_examples)
                elif is_gsm_mc:
                    prompt = build_gsm_mc_fewshot_prompt(q, choices, fewshot_examples)
                elif is_math_mc:
                    prompt = build_math_mc_fewshot_prompt(q, choices, fewshot_examples)
                elif is_openbookqa:
                    prompt = build_openbookqa_fewshot_prompt(q, choices, fewshot_examples)
                else:
                    prompt = build_prompt(q, choices, args.prompt_format, None)
            else:
                # lm-eval style: don't show options, score answer text
                # Exception: CommonsenseQA uses letter scoring in lm-harness
                if is_arc:
                    prompt = build_arc_fewshot_prompt_lmeval(q, choices, fewshot_examples)
                elif is_hellaswag:
                    prompt = build_hellaswag_fewshot_prompt_lmeval(q, choices, fewshot_examples)
                elif is_siqa:
                    prompt = build_siqa_fewshot_prompt_lmeval(q, choices, fewshot_examples)
                elif is_gsm_mc:
                    prompt = build_gsm_mc_fewshot_prompt_lmeval(q, choices, fewshot_examples)
                elif is_math_mc:
                    prompt = build_math_mc_fewshot_prompt_lmeval(q, choices, fewshot_examples)
                elif is_openbookqa:
                    prompt = build_openbookqa_fewshot_prompt_lmeval(q, choices, fewshot_examples)
                elif is_commonsenseqa:
                    # CommonsenseQA uses LETTER scoring in lm-harness (shows options)
                    prompt = build_commonsenseqa_fewshot_prompt(q, choices, fewshot_examples)
                elif is_sciq:
                    prompt = build_sciq_fewshot_prompt_lmeval(q, choices, fewshot_examples, support=sciq_support)
                elif is_race:
                    prompt = build_race_fewshot_prompt_lmeval(q, choices, fewshot_examples, article=race_article)
                else:
                    prompt = build_prompt(q, choices, args.prompt_format, None)
        else:
            prompt = build_prompt(q, choices, args.prompt_format, None)
        
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

        # Base scores + features
        # Scoring mode determines what we score:
        # - "lmeval": score answer TEXT (matches lm-eval-harness for ARC/HellaSwag/etc)
        # - "letter": score LETTER token (MMLU-style, makes ECE comparable across benchmarks)
        scores = []
        per_option_feats = []
        for opt_idx, c in enumerate(choices):
            if use_letter_scoring or is_mmlu or is_commonsenseqa:
                # Letter scoring: score just the letter token
                # CommonsenseQA uses letter scoring in lm-harness
                ans_text = f" {LETTERS[opt_idx]}"
            else:
                # lm-eval style: score the full answer text
                ans_text = f" {c}"
            ans_ids = tok(ans_text, add_special_tokens=False)["input_ids"]
            lp, feats = option_logprob_and_hidden(
                model, tok, prompt_ids, ans_ids, union_layers, pool=args.pool
            )
            scores.append(lp)
            per_option_feats.append(feats)

        p_base = softmax_np(np.array(scores, dtype=np.float64))       # (4,)
        pred_base = int(p_base.argmax())
        conf_b = float(p_base[pred_base])

        # Use original dataset index for question ID
        orig_idx = original_indices[idx] if original_indices else idx
        q_id = f"{args.dataset.split('/')[-1]}_{subject}_{args.split}_{orig_idx:05d}"

        # --- Steering with MLP ---
        # Build features for all 4 options
        hidden_concat = np.stack([probe._build_hidden_concat(feats_dict) for feats_dict in per_option_feats], axis=0)
        
        # Build option IDs for feature augmentation
        option_ids = np.array([f"{q_id}_opt{i}" for i in range(len(choices))])
        
        # Get MLP predictions (with feature augmentation)
        r_hat = probe.predict_residuals(hidden_concat, option_ids, p_base)  # (4,)
        
        # Optional clipping
        if args.residual_clip and args.residual_clip > 0:
            r_hat = np.clip(r_hat, -float(args.residual_clip), float(args.residual_clip))
        
        # Apply residual correction
        p_steer = residual_correction(p_base, r_hat, gamma=args.gamma)
        debug_payload = {"r_hat": r_hat.tolist()}

        pred_steer = int(p_steer.argmax())
        conf_s = float(p_steer[pred_steer])

        # Record
        acc_base.append(1 if pred_base == gold_idx else 0)
        acc_steer.append(1 if pred_steer == gold_idx else 0)
        conf_base.append(conf_b)
        conf_steer.append(conf_s)
        probs_base.append(p_base)
        probs_steer.append(p_steer)
        gold_labels.append(gold_idx)

        f_base.write(json.dumps({
            "id": q_id,
            "probs": [float(x) for x in p_base],
            "gold": gold_idx,
            "pred": pred_base,
            "conf": conf_b
        }) + "\n")

        out_steer = {
            "id": q_id,
            "probs": [float(x) for x in p_steer],
            "gold": gold_idx,
            "pred": pred_steer,
            "conf": conf_s
        }
        out_steer.update(debug_payload)
        f_steer.write(json.dumps(out_steer) + "\n")

        if (idx + 1) % 25 == 0:
            print(f"[{idx+1}/{len(ds)}] base_acc={np.mean(acc_base):.3f}  steer_acc={np.mean(acc_steer):.3f}")

    f_base.close(); f_steer.close()

    # Metrics (question-level, predicted-class confidence)
    base_conf_arr  = np.array(conf_base,  dtype=np.float64)
    steer_conf_arr = np.array(conf_steer, dtype=np.float64)
    corr_base  = np.array(acc_base,  dtype=np.int32)
    corr_steer = np.array(acc_steer, dtype=np.int32)

    # Compute cwECE
    cwece_base = compute_classwise_ece(probs_base, gold_labels, num_bins=25)
    cwece_steer = compute_classwise_ece(probs_steer, gold_labels, num_bins=25)

    metrics = {
        "baseline": {
            "Accuracy": float(np.mean(corr_base)),
            "ECE": ece_binary(base_conf_arr, corr_base),
            "Brier": brier_binary(base_conf_arr, corr_base),
            "NLL": nll_binary(base_conf_arr, corr_base),
            "cwECE": cwece_base["cwECE"],
            "Per_class_ECE": cwece_base["Per_class_ECE"],
        },
        "steered": {
            "Accuracy": float(np.mean(corr_steer)),
            "ECE": ece_binary(steer_conf_arr, corr_steer),
            "Brier": brier_binary(steer_conf_arr, corr_steer),
            "NLL": nll_binary(steer_conf_arr, corr_steer),
            "cwECE": cwece_steer["cwECE"],
            "Per_class_ECE": cwece_steer["Per_class_ECE"],
        }
    }

    # Custom JSON encoder to preserve precision
    class HighPrecisionEncoder(json.JSONEncoder):
        def iterencode(self, o, _one_shot=False):
            """Encode while preserving floating point precision."""
            if isinstance(o, float):
                # Format floats with high precision
                yield format(o, '.15g')
            else:
                yield from super().iterencode(o, _one_shot)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n==== RESULTS ====")
    # Print with more precision for better readability
    print("{")
    for condition in ["baseline", "steered"]:
        print(f'  "{condition}": {{')
        m = metrics[condition]
        print(f'    "Accuracy": {m["Accuracy"]:.10f},')
        print(f'    "ECE": {m["ECE"]:.10f},')
        print(f'    "Brier": {m["Brier"]:.10f},')
        print(f'    "NLL": {m["NLL"]:.10f},')
        print(f'    "cwECE": {m["cwECE"]:.10f},')
        print(f'    "Per_class_ECE": {m["Per_class_ECE"]}')
        if condition == "baseline":
            print("  },")
        else:
            print("  }")
    print("}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    # Data / model
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", type=str, default="allenai/openbookqa")
    ap.add_argument("--subset", type=str, default="main")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all")
    ap.add_argument("--pool", choices=["answer_mean","answer_final"], default="answer_mean")
    ap.add_argument("--dtype", type=str, default="auto", help="auto|bfloat16|float16|float32")

    # MLP Probe
    ap.add_argument("--probe_pkl", type=str, required=True,
                    help="Path to trained MLP probe pickle")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device for MLP inference (cuda/cpu)")

    # Combiner control
    ap.add_argument("--combiner", type=str, default="residual",
                    help="Steering method (only residual supported for MLP)")

    # Residual steering params
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Residual correction strength γ (0..1).")
    ap.add_argument("--residual_clip", type=float, default=0.0,
                    help="Optional |r̂|-clamp for stability (0 = off).")

    # IO
    ap.add_argument("--out_dir", type=str, required=True)
    
    # Train/val/test split control
    ap.add_argument("--split_ids_dir", type=str, default=None,
                    help="Directory containing test_row_indices.npy for 60/20/20 split. "
                         "If provided, only test questions will be used for steering.")
    
    # Prompt format control
    ap.add_argument("--prompt_format", type=str, default="legacy",
                    choices=["legacy", "lmeval"],
                    help="Prompt format: 'legacy' (Q: A)) or 'lmeval' (A.) to match activation collection.")
    ap.add_argument("--num_fewshot", type=int, default=0,
                    help="Number of few-shot examples (0 = zero-shot). Must match activation collection setting.")
    ap.add_argument("--scoring_mode", type=str, default="letter",
                    choices=["letter", "lmeval"],
                    help="Scoring mode: 'letter' (score A/B/C/D token) or 'lmeval' (score full answer text)")
    
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
