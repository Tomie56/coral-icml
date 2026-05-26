#!/usr/bin/env python3
"""
train_mlp_probe.py

Train MLP probes to predict calibration residuals from hidden states.
Parallel to train_linear_probe.py but uses neural networks instead of Ridge regression.

Usage:
    python src/train_mlp_probe.py \\
        --features_npz runs/deepseek-7b-chat-mmlu/probe_data.npz \\
        --layers 3 6 9 12 15 18 21 24 27 30 \\
        --out_dir runs/deepseek-7b-chat-mmlu/MLP \\
        --split_ids_dir runs/deepseek-7b-chat-mmlu
"""

import argparse
import os
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

from mlp_config import MLP_CONFIG


# ----------------------------- Feature Augmentation -----------------------------

def build_groups_from_ids(ids: np.ndarray) -> np.ndarray:
    """Extract question ID by removing _opt suffix."""
    return np.array([str(s).rsplit("_opt", 1)[0] for s in ids], dtype=object)


def augment_within_q_features(X: np.ndarray, ids: np.ndarray,
                              center_within_q: bool,
                              add_compete: bool) -> np.ndarray:
    """
    Returns augmented features (same as ridge regression):
      - If center_within_q: x' = x - mean(x over the 4 options of the same question)
      - If add_compete:     concat [x' (or x if not centered), (x - avg_others)]
    """
    from collections import defaultdict
    if not (center_within_q or add_compete):
        return X

    qids = build_groups_from_ids(ids)
    buckets = defaultdict(list)
    for i, q in enumerate(qids):
        buckets[q].append(i)

    X_base = X.copy()
    if center_within_q:
        for q, idxs in buckets.items():
            I = np.asarray(idxs, dtype=int)
            X_base[I] -= X_base[I].mean(axis=0, keepdims=True)

    if not add_compete:
        return X_base

    X_comp = np.zeros_like(X, dtype=X.dtype)
    for q, idxs in buckets.items():
        I = np.asarray(idxs, dtype=int)
        S = X[I].sum(axis=0, keepdims=True)
        m = len(I)
        for i in I:
            avg_others = (S - X[i]) / max(m - 1, 1)
            X_comp[i] = X[i] - avg_others

    X_aug = np.concatenate([X_base, X_comp], axis=1)
    return X_aug


def build_rank_feats(
    q_to_probs: Dict[str, np.ndarray],
    qid_per_row: np.ndarray,
    option_idx: np.ndarray,
) -> np.ndarray:
    """
    Compute per-option rank (1..n_choices) where 1 is highest probability.
    Returns (N, 1) float32.
    """
    feats = np.zeros((len(qid_per_row), 1), dtype=np.float32)
    for i, qid in enumerate(qid_per_row):
        probs = q_to_probs[qid]
        ranks = (-probs).argsort().argsort() + 1  # 1 = highest prob
        j = int(option_idx[i])
        if j < len(ranks):
            feats[i, 0] = float(ranks[j])
    return feats


def reconstruct_question_probs_from_rows(
    ids: np.ndarray,
    option_idx: np.ndarray,
    conf: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct per-question probability vectors from per-row data (variable choices).

    Returns:
      q_to_probs: qid -> (n_choices,) probs
      pred_idx_per_row: (N,) argmax option index for each row
      qid_per_row: (N,) question id string (without _opt)
      n_choices_per_row: (N,) number of choices for that question
    """
    from collections import defaultdict

    q_to_rows = defaultdict(list)
    qid_per_row = np.array([str(s).rsplit("_opt", 1)[0] for s in ids], dtype=object)
    for i, qid in enumerate(qid_per_row):
        q_to_rows[qid].append(i)

    q_to_probs: Dict[str, np.ndarray] = {}
    pred_idx_per_row = np.zeros(len(ids), dtype=np.int32)
    n_choices_per_row = np.zeros(len(ids), dtype=np.int32)

    for qid, row_indices in q_to_rows.items():
        I = np.asarray(row_indices, dtype=int)
        n_choices = int(option_idx[I].max()) + 1
        probs = np.zeros(n_choices, dtype=np.float32)
        for row_i in I:
            j = int(option_idx[row_i])
            if j < n_choices:
                probs[j] = float(conf[row_i])
        q_to_probs[qid] = probs
        pred_j = int(np.argmax(probs))
        pred_idx_per_row[I] = pred_j
        n_choices_per_row[I] = n_choices

    return q_to_probs, pred_idx_per_row, qid_per_row, n_choices_per_row


def build_logit_feature_block(
    q_to_probs: Dict[str, np.ndarray],
    qid_per_row: np.ndarray,
    option_idx: np.ndarray,
    pred_idx_per_row: np.ndarray,
    n_choices_per_row: np.ndarray,
    *,
    use_logp: bool,
    use_is_pred: bool,
    use_rank: bool,
    temperature_scales: List[float],
    use_temp_scaled: bool,
    use_pmax: bool,
    use_margin: bool,
    use_entropy: bool,
    use_centered_p: bool,
    use_compete: bool,
    use_option_onehot: bool,
) -> np.ndarray:
    """Build per-row logit/probability-derived features (baseline-style)."""
    N = len(option_idx)
    feats: List[List[float]] = []

    max_choices = int(n_choices_per_row.max()) if n_choices_per_row is not None and len(n_choices_per_row) > 0 else 4

    temps = [float(t) for t in temperature_scales]
    temps_extra = [t for t in temps if abs(t - 1.0) > 1e-8]

    for i in range(N):
        qid = qid_per_row[i]
        probs = q_to_probs[qid]
        j = int(option_idx[i])
        pred_j = int(pred_idx_per_row[i])
        n_choices = int(n_choices_per_row[i])

        p_j = float(probs[j])
        row: List[float] = []

        # Always include p_j when logit feature block is enabled
        row.append(p_j)

        if use_logp:
            row.append(float(np.log(max(p_j, 1e-12))))

        if use_is_pred:
            row.append(1.0 if j == pred_j else 0.0)

        if use_rank:
            # 1..4 (ties get half-ranks)
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
            if j < n_choices:
                onehot[j] = 1.0
            row.extend(onehot)

        feats.append(row)

    return np.asarray(feats, dtype=np.float32)


def apply_feature_pipeline(
    X_rows: np.ndarray,
    ids: np.ndarray,
    option_idx: np.ndarray,
    conf: np.ndarray,
    *,
    use_center_within_q: bool,
    use_compete_act: bool,
    use_rankfeat: bool,
    use_rank_onehot: bool,
    use_logit_block: bool,
    logit_use_logp: bool,
    logit_use_is_pred: bool,
    logit_use_rank: bool,
    logit_use_temp_scaled: bool,
    logit_temperature_scales: List[float],
    logit_use_pmax: bool,
    logit_use_margin: bool,
    logit_use_entropy: bool,
    logit_use_centered_p: bool,
    logit_use_compete: bool,
    logit_use_option_onehot: bool,
) -> np.ndarray:
    """Apply activation + optional baseline-logit feature augmentation."""
    X_aug = augment_within_q_features(
        X_rows,
        ids,
        center_within_q=use_center_within_q,
        add_compete=use_compete_act,
    )

    q_to_probs = None
    pred_idx_per_row = None
    qid_per_row = build_groups_from_ids(ids)
    n_choices_per_row = None

    if use_rankfeat or use_logit_block:
        q_to_probs, pred_idx_per_row, qid_per_row, n_choices_per_row = reconstruct_question_probs_from_rows(
            ids=ids,
            option_idx=option_idx,
            conf=conf,
        )

    if use_rankfeat and q_to_probs is not None:
        X_aug = np.concatenate([X_aug, build_rank_feats(q_to_probs, qid_per_row, option_idx)], axis=1)
    if use_rank_onehot:
        X_aug = np.concatenate([X_aug, build_rank_onehot_feats(conf, ids)], axis=1)

    if use_logit_block:
        logit_feats = build_logit_feature_block(
            q_to_probs,
            qid_per_row,
            option_idx,
            pred_idx_per_row,
            n_choices_per_row,
            use_logp=logit_use_logp,
            use_is_pred=logit_use_is_pred,
            use_rank=logit_use_rank,
            temperature_scales=logit_temperature_scales,
            use_temp_scaled=logit_use_temp_scaled,
            use_pmax=logit_use_pmax,
            use_margin=logit_use_margin,
            use_entropy=logit_use_entropy,
            use_centered_p=logit_use_centered_p,
            use_compete=logit_use_compete,
            use_option_onehot=logit_use_option_onehot,
        )
        X_aug = np.concatenate([X_aug, logit_feats], axis=1)

    return X_aug


def build_rank_onehot_feats(probs: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """
    Compute one-hot encoding of per-option rank (1..max_choices).
    Returns (N, max_choices-1) float32 with columns for rank==2, rank==3, ..., rank==max_choices.
    Rank==1 is the implicit baseline (all zeros).
    Handles variable number of choices per question.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, g in enumerate(group_ids):
        buckets[g].append(i)
    
    # Find max choices across all questions
    max_choices = max(len(idxs) for idxs in buckets.values())
    
    # Columns for rank 2, 3, ..., max_choices (rank 1 is implicit baseline)
    feats = np.zeros((len(probs), max_choices - 1), dtype=np.float32)
    for g, idxs in buckets.items():
        I = np.asarray(idxs, dtype=int)
        p = probs[I]
        ranks = (-p).argsort().argsort() + 1
        for j, i in enumerate(I):
            r = int(ranks[j])
            if 2 <= r <= max_choices:
                feats[i, r - 2] = 1.0  # rank 2 -> col 0, rank 3 -> col 1, etc.
    return feats


# ----------------------------- MLP Model -----------------------------

class MLPProbe(nn.Module):
    """Multi-layer perceptron for residual prediction."""
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
            
            # Activation
            if config["activation"] == "relu":
                layers.append(nn.ReLU())
            elif config["activation"] == "gelu":
                layers.append(nn.GELU())
            elif config["activation"] == "tanh":
                layers.append(nn.Tanh())
            
            # Dropout
            if config.get("dropout", 0.0) > 0:
                layers.append(nn.Dropout(config["dropout"]))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config["output_dim"]))
        
        # Output activation (optional, for bounded outputs)
        self.output_activation = config.get("output_activation")
        self.output_scale = config.get("output_scale", 1.0)
        
        if self.output_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        # If None or not specified, output is unbounded
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.network(x).squeeze(-1)  # (batch,)
        
        # Apply output scaling if using bounded activation
        if self.output_activation in ["tanh", "sigmoid"]:
            output = output * self.output_scale
        
        return output


def find_layer_matrix(store: Dict[str, np.ndarray], L: int) -> np.ndarray:
    """Find layer activations in NPZ store."""
    for key in (f"layer_{L}", f"h_{L}"):
        if key in store:
            return store[key]
    raise KeyError(f"No features found for layer {L}")


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for z-score normalization."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.maximum(sigma, 1e-12)
    return mu, sigma


def zscore_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply z-score normalization."""
    return (X - mu) / sigma


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    
    # Pearson correlation
    if len(y_true) > 1:
        corr, _ = pearsonr(y_true, y_pred)
    else:
        corr = 0.0
    
    return {
        "R2": float(r2),
        "MAE": mae,
        "RMSE": rmse,
        "Pearson_r": float(corr),
    }


def train_mlp_single_config(
    X_train_norm: np.ndarray,
    y_train: np.ndarray,
    X_val_norm: np.ndarray,
    y_val: np.ndarray,
    weight_decay: float,
    output_penalty: float,
    config_base: Dict,
    device: str = "cuda"
) -> Dict:
    """
    Train MLP with specific weight_decay and output_penalty values.
    
    Returns:
        Dictionary with validation metrics and trained model
    """
    config = config_base.copy()
    config["output_penalty"] = output_penalty  # Set for this trial
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Initialize model
    model = MLPProbe(config).to(device)
    
    # Loss and optimizer
    if config["loss_fn"] == "mse":
        criterion = nn.MSELoss()
    elif config["loss_fn"] == "mae":
        criterion = nn.L1Loss()
    elif config["loss_fn"] == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay)
    elif config["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay, momentum=0.9)
    
    # Learning rate scheduler
    if config.get("lr_scheduler") == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.get("lr_factor", 0.5),
            patience=config.get("lr_patience", 5), verbose=False
        )
    else:
        scheduler = None
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get("patience", 30)
    
    for epoch in range(config["num_epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Add output penalty to discourage extreme predictions
            if config.get("output_penalty", 0.0) > 0:
                output_penalty = config["output_penalty"] * torch.mean(outputs ** 2)
                loss = loss + output_penalty
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_X)
        
        train_loss /= len(y_train)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Add output penalty (for tracking consistency)
                if config.get("output_penalty", 0.0) > 0:
                    output_penalty = config["output_penalty"] * torch.mean(outputs ** 2)
                    loss = loss + output_penalty
                
                val_loss += loss.item() * len(batch_X)
                val_preds.append(outputs.cpu().numpy())
        
        val_loss /= len(y_val)
        val_preds = np.concatenate(val_preds)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            best_val_preds = val_preds
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final validation metrics
    val_metrics = metrics_regression(y_val, best_val_preds)
    
    return {
        "model_state": best_model_state,
        "val_metrics": val_metrics,
        "val_loss": best_val_loss,
    }


def train_mlp_single_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ids_train: np.ndarray,
    p_hat_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ids_val: np.ndarray,
    p_hat_val: np.ndarray,
    layer_idx: int,
    config: Dict,
    device: str = "cuda",
    *,
    feature_flags: Dict
) -> Dict:
    """
    Train MLP probe for a single layer with weight_decay grid search.
    
    Args:
        X_train: Raw training features (N_train, D_raw)
        y_train: Training targets (N_train,)
        ids_train: Training question IDs - for feature augmentation
        p_hat_train: Training base model probabilities - for rank features
        X_val: Raw validation features (N_val, D_raw)
        y_val: Validation targets (N_val,)
        ids_val: Validation question IDs
        p_hat_val: Validation base model probabilities
        layer_idx: Layer index for logging
        config: MLP configuration (with weight_decay_grid)
        device: Device for training
    
    Returns:
        Dictionary with best trained model, normalization params, and validation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training MLP Probe for Layer {layer_idx}")
    print(f"{'='*60}")
    print(f"Train: N={len(X_train)}, D_raw={X_train.shape[1]}")
    print(f"Val:   N={len(X_val)}, D_raw={X_val.shape[1]}")
    
    print("Applying feature augmentation...")
    X_train_aug = apply_feature_pipeline(
        X_train,
        ids_train,
        option_idx=feature_flags["option_idx_train"],
        conf=p_hat_train,
        use_center_within_q=bool(feature_flags["use_center_within_q"]),
        use_compete_act=bool(feature_flags["use_compete"]),
        use_rankfeat=bool(feature_flags["use_rankfeat"]),
        use_rank_onehot=bool(feature_flags["use_rank_onehot"]),
        use_logit_block=bool(feature_flags["use_logit_block"]),
        logit_use_logp=bool(feature_flags["logit_use_logp"]),
        logit_use_is_pred=bool(feature_flags["logit_use_is_pred"]),
        logit_use_rank=bool(feature_flags["logit_use_rank"]),
        logit_use_temp_scaled=bool(feature_flags["logit_use_temp_scaled"]),
        logit_temperature_scales=list(feature_flags["logit_temperature_scales"]),
        logit_use_pmax=bool(feature_flags["logit_use_pmax"]),
        logit_use_margin=bool(feature_flags["logit_use_margin"]),
        logit_use_entropy=bool(feature_flags["logit_use_entropy"]),
        logit_use_centered_p=bool(feature_flags["logit_use_centered_p"]),
        logit_use_compete=bool(feature_flags["logit_use_compete"]),
        logit_use_option_onehot=bool(feature_flags["logit_use_option_onehot"]),
    )

    X_val_aug = apply_feature_pipeline(
        X_val,
        ids_val,
        option_idx=feature_flags["option_idx_val"],
        conf=p_hat_val,
        use_center_within_q=bool(feature_flags["use_center_within_q"]),
        use_compete_act=bool(feature_flags["use_compete"]),
        use_rankfeat=bool(feature_flags["use_rankfeat"]),
        use_rank_onehot=bool(feature_flags["use_rank_onehot"]),
        use_logit_block=bool(feature_flags["use_logit_block"]),
        logit_use_logp=bool(feature_flags["logit_use_logp"]),
        logit_use_is_pred=bool(feature_flags["logit_use_is_pred"]),
        logit_use_rank=bool(feature_flags["logit_use_rank"]),
        logit_use_temp_scaled=bool(feature_flags["logit_use_temp_scaled"]),
        logit_temperature_scales=list(feature_flags["logit_temperature_scales"]),
        logit_use_pmax=bool(feature_flags["logit_use_pmax"]),
        logit_use_margin=bool(feature_flags["logit_use_margin"]),
        logit_use_entropy=bool(feature_flags["logit_use_entropy"]),
        logit_use_centered_p=bool(feature_flags["logit_use_centered_p"]),
        logit_use_compete=bool(feature_flags["logit_use_compete"]),
        logit_use_option_onehot=bool(feature_flags["logit_use_option_onehot"]),
    )
    
    print(f"  After augmentation: D_final={X_train_aug.shape[1]}")
    
    # Update config input dim to match augmented features
    config_updated = config.copy()
    config_updated["input_dim"] = X_train_aug.shape[1]
    
    print(f"Config: {config_updated['hidden_dims']}, dropout={config_updated['dropout']}, lr={config_updated['learning_rate']}")
    
    # Z-score normalization (fit on TRAIN, apply to both)
    mu, sigma = zscore_fit(X_train_aug)
    X_train_norm = zscore_apply(X_train_aug, mu, sigma)
    X_val_norm = zscore_apply(X_val_aug, mu, sigma)
    
    # Grid search over weight_decay and output_penalty
    weight_decay_grid = config_updated.get("weight_decay_grid", [1.0, 2.0, 5.0, 10.0, 20.0])
    output_penalty_grid = config_updated.get("output_penalty_grid", [0.0, 0.001, 0.01, 0.1])
    
    print(f"\nGrid searching:")
    print(f"  weight_decay: {weight_decay_grid}")
    print(f"  output_penalty: {output_penalty_grid}")
    print(f"  Total trials: {len(weight_decay_grid) * len(output_penalty_grid)}")
    
    best_wd = None
    best_op = None
    best_metrics = None
    best_model_state = None
    best_val_r2 = -float('inf')
    
    grid_results = []
    for wd in weight_decay_grid:
        for op in output_penalty_grid:
            print(f"  Testing weight_decay={wd}, output_penalty={op}...")
            result = train_mlp_single_config(
                X_train_norm, y_train, X_val_norm, y_val,
                weight_decay=wd, output_penalty=op, config_base=config_updated, device=device
            )
            
            val_r2 = result["val_metrics"]["R2"]
            grid_results.append({
                "weight_decay": wd,
                "output_penalty": op,
                "val_R2": val_r2,
                "val_MAE": result["val_metrics"]["MAE"],
                "val_RMSE": result["val_metrics"]["RMSE"],
            })
            print(f"    → R²={val_r2:.4f}, MAE={result['val_metrics']['MAE']:.4f}")
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_wd = wd
                best_op = op
                best_metrics = result["val_metrics"]
                best_model_state = result["model_state"]
    
    print(f"\n  BEST: weight_decay={best_wd}, output_penalty={best_op} → R²={best_val_r2:.4f}")
    
    # Load best model
    model = MLPProbe(config_updated).to(device)
    model.load_state_dict(best_model_state)
    
    print("Training complete!")
    
    # Return artifact
    return {
        "model_state_dict": model.state_dict(),
        "mu": mu,
        "sigma": sigma,
        "config": config_updated,  # Store updated config with correct input_dim
        "layer_index": layer_idx,
        "val_metrics": best_metrics,
        "best_weight_decay": best_wd,
        "best_output_penalty": best_op,
        "grid_search_results": grid_results,
        "mode": "residual_prob",
        "meta": {
            "use_center_within_q": bool(feature_flags["use_center_within_q"]),
            "use_compete": bool(feature_flags["use_compete"]),
            "use_rankfeat": bool(feature_flags["use_rankfeat"]),
            "use_rank_onehot": bool(feature_flags["use_rank_onehot"]),
            "use_logit_block": bool(feature_flags["use_logit_block"]),
            "logit_temperature_scales": list(feature_flags["logit_temperature_scales"]),
            "logit_use_logp": bool(feature_flags["logit_use_logp"]),
            "logit_use_is_pred": bool(feature_flags["logit_use_is_pred"]),
            "logit_use_rank": bool(feature_flags["logit_use_rank"]),
            "logit_use_temp_scaled": bool(feature_flags["logit_use_temp_scaled"]),
            "logit_use_pmax": bool(feature_flags["logit_use_pmax"]),
            "logit_use_margin": bool(feature_flags["logit_use_margin"]),
            "logit_use_entropy": bool(feature_flags["logit_use_entropy"]),
            "logit_use_centered_p": bool(feature_flags["logit_use_centered_p"]),
            "logit_use_compete": bool(feature_flags["logit_use_compete"]),
            "logit_use_option_onehot": bool(feature_flags["logit_use_option_onehot"]),
        },
    }

def train_mlp_concat_layers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ids_train: np.ndarray,
    p_hat_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ids_val: np.ndarray,
    p_hat_val: np.ndarray,
    layer_indices: List[int],
    layer_dims: List[int],
    config: Dict,
    device: str = "cuda"
) -> Dict:
    """
    Train a *single* MLP probe on concatenated features from multiple layers.
    This mirrors train_mlp_single_layer but treats all requested layers as one
    big feature vector.
    """
    print(f"\n{'='*60}")
    print(f"Training CONCAT MLP Probe for layers {layer_indices}")
    print(f"{'='*60}")
    print(f"Train: N={len(X_train)}, D_raw={X_train.shape[1]}")
    print(f"Val:   N={len(X_val)}, D_raw={X_val.shape[1]}")
    print(f"  Per-layer dims: {layer_dims}")
    
    print("Applying feature augmentation (concat)...")
    X_train_aug = apply_feature_pipeline(
        X_train,
        ids_train,
        option_idx=feature_flags["option_idx_train"],
        conf=p_hat_train,
        use_center_within_q=bool(feature_flags["use_center_within_q"]),
        use_compete_act=bool(feature_flags["use_compete"]),
        use_rankfeat=bool(feature_flags["use_rankfeat"]),
        use_rank_onehot=bool(feature_flags["use_rank_onehot"]),
        use_logit_block=bool(feature_flags["use_logit_block"]),
        logit_use_logp=bool(feature_flags["logit_use_logp"]),
        logit_use_is_pred=bool(feature_flags["logit_use_is_pred"]),
        logit_use_rank=bool(feature_flags["logit_use_rank"]),
        logit_use_temp_scaled=bool(feature_flags["logit_use_temp_scaled"]),
        logit_temperature_scales=list(feature_flags["logit_temperature_scales"]),
        logit_use_pmax=bool(feature_flags["logit_use_pmax"]),
        logit_use_margin=bool(feature_flags["logit_use_margin"]),
        logit_use_entropy=bool(feature_flags["logit_use_entropy"]),
        logit_use_centered_p=bool(feature_flags["logit_use_centered_p"]),
        logit_use_compete=bool(feature_flags["logit_use_compete"]),
        logit_use_option_onehot=bool(feature_flags["logit_use_option_onehot"]),
    )

    X_val_aug = apply_feature_pipeline(
        X_val,
        ids_val,
        option_idx=feature_flags["option_idx_val"],
        conf=p_hat_val,
        use_center_within_q=bool(feature_flags["use_center_within_q"]),
        use_compete_act=bool(feature_flags["use_compete"]),
        use_rankfeat=bool(feature_flags["use_rankfeat"]),
        use_rank_onehot=bool(feature_flags["use_rank_onehot"]),
        use_logit_block=bool(feature_flags["use_logit_block"]),
        logit_use_logp=bool(feature_flags["logit_use_logp"]),
        logit_use_is_pred=bool(feature_flags["logit_use_is_pred"]),
        logit_use_rank=bool(feature_flags["logit_use_rank"]),
        logit_use_temp_scaled=bool(feature_flags["logit_use_temp_scaled"]),
        logit_temperature_scales=list(feature_flags["logit_temperature_scales"]),
        logit_use_pmax=bool(feature_flags["logit_use_pmax"]),
        logit_use_margin=bool(feature_flags["logit_use_margin"]),
        logit_use_entropy=bool(feature_flags["logit_use_entropy"]),
        logit_use_centered_p=bool(feature_flags["logit_use_centered_p"]),
        logit_use_compete=bool(feature_flags["logit_use_compete"]),
        logit_use_option_onehot=bool(feature_flags["logit_use_option_onehot"]),
    )
    
    print(f"  After augmentation: D_final={X_train_aug.shape[1]}")
    
    # Update config input dim to match augmented features
    config_updated = config.copy()
    config_updated["input_dim"] = X_train_aug.shape[1]
    
    print(f"Config: {config_updated['hidden_dims']}, dropout={config_updated['dropout']}, lr={config_updated['learning_rate']}")
    
    # Z-score normalization (fit on TRAIN, apply to both)
    mu, sigma = zscore_fit(X_train_aug)
    X_train_norm = zscore_apply(X_train_aug, mu, sigma)
    X_val_norm = zscore_apply(X_val_aug, mu, sigma)
    
    # Grid search over weight_decay and output_penalty
    weight_decay_grid = config_updated.get("weight_decay_grid", [1.0, 2.0, 5.0, 10.0, 20.0])
    output_penalty_grid = config_updated.get("output_penalty_grid", [0.0, 0.001, 0.01, 0.1])
    
    print(f"\nGrid searching (concat):")
    print(f"  weight_decay: {weight_decay_grid}")
    print(f"  output_penalty: {output_penalty_grid}")
    print(f"  Total trials: {len(weight_decay_grid) * len(output_penalty_grid)}")
    
    best_wd = None
    best_op = None
    best_metrics = None
    best_model_state = None
    best_val_r2 = -float('inf')
    
    grid_results = []
    for wd in weight_decay_grid:
        for op in output_penalty_grid:
            print(f"  Testing weight_decay={wd}, output_penalty={op}...")
            result = train_mlp_single_config(
                X_train_norm, y_train, X_val_norm, y_val,
                weight_decay=wd, output_penalty=op, config_base=config_updated, device=device
            )
            
            val_r2 = result["val_metrics"]["R2"]
            grid_results.append({
                "weight_decay": wd,
                "output_penalty": op,
                "val_R2": val_r2,
                "val_MAE": result["val_metrics"]["MAE"],
                "val_RMSE": result["val_metrics"]["RMSE"],
            })
            print(f"    → R²={val_r2:.4f}, MAE={result['val_metrics']['MAE']:.4f}")
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_wd = wd
                best_op = op
                best_metrics = result["val_metrics"]
                best_model_state = result["model_state"]
    
    print(f"\n  BEST (concat): weight_decay={best_wd}, output_penalty={best_op} → R²={best_val_r2:.4f}")
    
    # Load best model
    model = MLPProbe(config_updated).to(device)
    model.load_state_dict(best_model_state)
    
    print("Training complete (concat)!")
    
    return {
        "model_state_dict": model.state_dict(),
        "mu": mu,
        "sigma": sigma,
        "config": config_updated,
        "layer_indices": list(layer_indices),
        "layer_dims": list(layer_dims),
        "val_metrics": best_metrics,
        "best_weight_decay": best_wd,
        "best_output_penalty": best_op,
        "grid_search_results": grid_results,
        "mode": "residual_prob",
        "meta": {
            "use_center_within_q": True,
            "use_compete": False,
            "use_rankfeat": True,
            "use_rank_onehot": True,
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Train MLP probes for calibration")
    parser.add_argument("--features_npz", required=True, help="Path to probe_data.npz")
    parser.add_argument("--layers", nargs="+", type=int, required=True, help="Layers to train")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--split_ids_dir", required=True, help="Directory with train/val_row_indices.npy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--concat_layers",
        action="store_true",
        help="If set, concatenate all requested layers and train a single MLP probe",
    )

    # Existing augmentation toggles (defaults match previous behavior)
    parser.add_argument("--center_within_q", dest="center_within_q", action="store_true")
    parser.add_argument("--no_center_within_q", dest="center_within_q", action="store_false")
    parser.set_defaults(center_within_q=True)

    parser.add_argument("--use_compete", action="store_true", default=False)

    parser.add_argument("--rankfeat", dest="rankfeat", action="store_true")
    parser.add_argument("--no_rankfeat", dest="rankfeat", action="store_false")
    parser.set_defaults(rankfeat=True)

    parser.add_argument("--rank_onehot", dest="rank_onehot", action="store_true")
    parser.add_argument("--no_rank_onehot", dest="rank_onehot", action="store_false")
    parser.set_defaults(rank_onehot=True)

    # Baseline logit/prob feature block toggles
    parser.add_argument("--use_logit_block", action="store_true", default=False)
    parser.add_argument("--logit_temperature_scales", type=str, default="0.7,1.0,1.3")
    parser.add_argument("--logit_use_logp", action="store_true", default=False)
    parser.add_argument("--logit_use_is_pred", action="store_true", default=False)
    parser.add_argument("--logit_use_rank", action="store_true", default=False)
    parser.add_argument("--logit_use_temp_scaled", action="store_true", default=False)
    parser.add_argument("--logit_use_pmax", action="store_true", default=False)
    parser.add_argument("--logit_use_margin", action="store_true", default=False)
    parser.add_argument("--logit_use_entropy", action="store_true", default=False)
    parser.add_argument("--logit_use_centered_p", action="store_true", default=False)
    parser.add_argument("--logit_use_compete", action="store_true", default=False)
    parser.add_argument("--logit_use_option_onehot", action="store_true", default=False)

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from {args.features_npz}")
    data = np.load(args.features_npz, allow_pickle=True)

    # Load train and val split indices
    split_dir = Path(args.split_ids_dir)
    train_indices_path = split_dir / "train_row_indices.npy"
    val_indices_path = split_dir / "val_row_indices.npy"

    if not train_indices_path.exists():
        raise FileNotFoundError(f"Train indices not found: {train_indices_path}")
    if not val_indices_path.exists():
        raise FileNotFoundError(f"Val indices not found: {val_indices_path}")

    train_indices = np.load(train_indices_path)
    val_indices = np.load(val_indices_path)
    print(f"Using train split: {len(train_indices)} rows")
    print(f"Using val split: {len(val_indices)} rows")

    # Get targets, IDs, and probabilities for feature augmentation
    y_all = data["residual_prob"].astype(np.float32)
    ids_all = data["ids"].astype(str)
    option_idx_all = data["option_idx"].astype(np.int32)
    # Use 'conf' (per-option probability) for rank features, NOT 'p_hat'
    # p_hat has inconsistent values: conf for predicted option, p_option for others
    # conf stores p_option for ALL rows, which is what we need for ranking
    conf_all = data["conf"].astype(np.float32)

    # Filter to train/val
    y_train = y_all[train_indices]
    ids_train = ids_all[train_indices]
    conf_train = conf_all[train_indices]
    option_idx_train = option_idx_all[train_indices]

    y_val = y_all[val_indices]
    ids_val = ids_all[val_indices]
    conf_val = conf_all[val_indices]
    option_idx_val = option_idx_all[val_indices]

    print("Target: residual_prob")
    print(f"  Train: N={len(y_train)}")
    print(f"  Val:   N={len(y_val)}")
    print(f"  Using 'conf' (per-option prob) for rank features")

    use_center_within_q = bool(args.center_within_q)
    use_rankfeat = bool(args.rankfeat)
    use_rank_onehot = bool(args.rank_onehot)
    logit_temperature_scales = [float(x) for x in args.logit_temperature_scales.split(",") if x.strip()]

    feature_flags_base = {
        "use_center_within_q": use_center_within_q,
        "use_compete": bool(args.use_compete),
        "use_rankfeat": use_rankfeat,
        "use_rank_onehot": use_rank_onehot,
        "use_logit_block": bool(args.use_logit_block),
        "logit_temperature_scales": logit_temperature_scales,
        "logit_use_logp": bool(args.logit_use_logp),
        "logit_use_is_pred": bool(args.logit_use_is_pred),
        "logit_use_rank": bool(args.logit_use_rank),
        "logit_use_temp_scaled": bool(args.logit_use_temp_scaled),
        "logit_use_pmax": bool(args.logit_use_pmax),
        "logit_use_margin": bool(args.logit_use_margin),
        "logit_use_entropy": bool(args.logit_use_entropy),
        "logit_use_centered_p": bool(args.logit_use_centered_p),
        "logit_use_compete": bool(args.logit_use_compete),
        "logit_use_option_onehot": bool(args.logit_use_option_onehot),
    }

    if args.concat_layers:
        # Build concatenated features for all requested layers
        X_list_all = []
        layer_dims = []
        for layer_idx in args.layers:
            X_all = find_layer_matrix(data, layer_idx)
            X_list_all.append(X_all)
            layer_dims.append(int(X_all.shape[1]))
        X_all_concat = np.concatenate(X_list_all, axis=1)
        X_train = X_all_concat[train_indices]
        X_val = X_all_concat[val_indices]

        artifact = train_mlp_concat_layers(
            X_train,
            y_train,
            ids_train,
            conf_train,
            X_val,
            y_val,
            ids_val,
            conf_val,
            layer_indices=list(args.layers),
            layer_dims=layer_dims,
            config=MLP_CONFIG,
            device=device,
            feature_flags={
                **feature_flags_base,
                "option_idx_train": option_idx_train,
                "option_idx_val": option_idx_val,
            },
        )

        out_path = out_dir / "Lconcat.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Saved concat probe to: {out_path}")

        summary_data = {
            "settings": {
                "config": MLP_CONFIG,
                "weight_decay_grid": MLP_CONFIG.get("weight_decay_grid", []),
                "output_penalty_grid": MLP_CONFIG.get("output_penalty_grid", []),
                "seed": args.seed,
                "using_train_val_split": True,
                "n_train_rows": len(y_train),
                "n_val_rows": len(y_val),
                "concat_layers": True,
                "layers": list(args.layers),
            },
            "reports": [
                {
                    "key": "concat",
                    "layer_indices": list(args.layers),
                    "layer_dims": layer_dims,
                    "val_R2": artifact["val_metrics"]["R2"],
                    "val_MAE": artifact["val_metrics"]["MAE"],
                    "val_RMSE": artifact["val_metrics"]["RMSE"],
                    "val_Pearson_r": artifact["val_metrics"]["Pearson_r"],
                    "best_weight_decay": artifact["best_weight_decay"],
                    "best_output_penalty": artifact["best_output_penalty"],
                    "grid_search_results": artifact["grid_search_results"],
                }
            ],
        }
        summary_data["best_by_R2"] = summary_data["reports"][0]

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n{'='*70}")
        print("Concat training complete!")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*70}")
        return

    # Per-layer training (original behavior)
    summary_data = {
        "settings": {
            "config": MLP_CONFIG,
            "weight_decay_grid": MLP_CONFIG.get("weight_decay_grid", []),
            "output_penalty_grid": MLP_CONFIG.get("output_penalty_grid", []),
            "seed": args.seed,
            "using_train_val_split": True,
            "n_train_rows": len(y_train),
            "n_val_rows": len(y_val),
        },
        "reports": [],
    }

    for layer_idx in args.layers:
        print(f"\n{'='*70}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'='*70}")

        try:
            X_all = find_layer_matrix(data, layer_idx)
            X_train = X_all[train_indices]
            X_val = X_all[val_indices]

            artifact = train_mlp_single_layer(
                X_train,
                y_train,
                ids_train,
                conf_train,
                X_val,
                y_val,
                ids_val,
                conf_val,
                layer_idx,
                MLP_CONFIG,
                device,
                feature_flags={
                    **feature_flags_base,
                    "option_idx_train": option_idx_train,
                    "option_idx_val": option_idx_val,
                },
            )

            out_path = out_dir / f"L{layer_idx}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(artifact, f)
            print(f"Saved to: {out_path}")

            summary_data["reports"].append(
                {
                    "key": f"L{layer_idx}",
                    "layer": layer_idx,
                    "val_R2": artifact["val_metrics"]["R2"],
                    "val_MAE": artifact["val_metrics"]["MAE"],
                    "val_RMSE": artifact["val_metrics"]["RMSE"],
                    "val_Pearson_r": artifact["val_metrics"]["Pearson_r"],
                    "best_weight_decay": artifact["best_weight_decay"],
                    "best_output_penalty": artifact["best_output_penalty"],
                    "grid_search_results": artifact["grid_search_results"],
                }
            )

        except Exception as e:
            print(f"ERROR processing layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()

    if summary_data["reports"]:
        best = max(summary_data["reports"], key=lambda x: x["val_R2"])
        summary_data["best_by_R2"] = best

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
