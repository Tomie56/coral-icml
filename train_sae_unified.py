#!/usr/bin/env python3
"""
Train Sparse Autoencoder (SAE) on ALL training data (no binning).

This script trains a single SAE on all training split activations to discover
general monosemantic features for circuit analysis.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from sae_config import create_sae


def train_sae(
    activations,
    sae,
    optimizer,
    n_epochs=100,
    batch_size=256,
    device='cuda',
    normalize_every=100,
    verbose=True,
):
    """
    Train the SAE on the given activations.
    
    Args:
        activations: (n_examples, d_model) numpy array
        sae: SAE model
        optimizer: PyTorch optimizer
        n_epochs: number of training epochs
        batch_size: batch size
        device: device to train on
        normalize_every: normalize decoder weights every N steps
        verbose: whether to print progress
    
    Returns:
        losses: dict with loss history and normalization stats
    """
    sae.train()
    n_examples = activations.shape[0]
    
    # Normalize activations (subtract mean, divide by std)
    # This is CRITICAL for SAE training!
    act_mean = activations.mean(axis=0)
    act_std = activations.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
    activations_normalized = (activations - act_mean) / act_std
    
    if verbose:
        print(f"Input normalization:")
        print(f"  Original - Mean: {activations.mean():.6f}, Std: {activations.std():.6f}")
        print(f"  Normalized - Mean: {activations_normalized.mean():.6f}, Std: {activations_normalized.std():.6f}")
        print()
    
    # Convert to torch tensor
    activations_tensor = torch.from_numpy(activations_normalized).float().to(device)
    
    # Training loop
    epoch_losses = []
    recon_losses = []
    sparsity_losses = []
    step = 0
    
    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(n_examples)
        activations_shuffled = activations_tensor[perm]
        
        batch_losses = []
        batch_recon = []
        batch_sparsity = []
        
        # Mini-batch training
        n_batches = (n_examples + batch_size - 1) // batch_size
        
        iterator = range(0, n_examples, batch_size)
        if verbose:
            iterator = tqdm(iterator, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        
        for i in iterator:
            batch = activations_shuffled[i:i+batch_size]
            
            # Forward pass
            optimizer.zero_grad()
            x_hat, f, loss_dict = sae(batch, compute_loss=True)
            
            # Backward pass
            loss_dict['total'].backward()
            optimizer.step()
            
            # Normalize decoder weights periodically
            step += 1
            if step % normalize_every == 0:
                sae.normalize_decoder_weights()
            
            batch_losses.append(loss_dict['total'].item())
            batch_recon.append(loss_dict['reconstruction'].item())
            batch_sparsity.append(loss_dict['sparsity'].item())
        
        avg_loss = np.mean(batch_losses)
        avg_recon = np.mean(batch_recon)
        avg_sparsity = np.mean(batch_sparsity)
        
        epoch_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        sparsity_losses.append(avg_sparsity)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Total={avg_loss:.6f}, Recon={avg_recon:.6f}, Sparsity={avg_sparsity:.6f}")
    
    return {
        'total': epoch_losses,
        'reconstruction': recon_losses,
        'sparsity': sparsity_losses,
        'normalization': {
            'mean': act_mean,
            'std': act_std,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train unified SAE on all training data (no binning)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to probe_data.npz file",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
        help="Directory containing train/val/test split indices",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for trained SAE",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Which layer to train SAE on",
    )
    parser.add_argument(
        "--d_sparse",
        type=int,
        default=16384,
        help="Sparse dimension (default: 16384, 4x expansion)",
    )
    parser.add_argument(
        "--lambda_sparsity",
        type=float,
        default=0.0001,
        help="Sparsity penalty weight (default: 0.0001)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("SAE TRAINING - UNIFIED (NO BINNING)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Split dir: {args.split_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Layer: {args.layer}")
    print(f"d_sparse: {args.d_sparse}")
    print(f"lambda_sparsity: {args.lambda_sparsity}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print("="*70)
    print()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading probe_data.npz...")
    data = np.load(args.data_path)
    
    # Load train indices
    split_dir = Path(args.split_dir)
    train_indices = np.load(split_dir / "train_row_indices.npy")
    
    # Get layer activations
    layer_key = f"layer_{args.layer}"
    all_activations = data[layer_key]  # (n_total, d_model)
    
    # Extract training activations
    train_activations = all_activations[train_indices]
    
    d_model = train_activations.shape[1]
    n_train = train_activations.shape[0]
    
    print(f"Train activations: {train_activations.shape}")
    print(f"d_model: {d_model}")
    print(f"n_train: {n_train:,}")
    print()
    
    # Create SAE
    print("Creating SAE...")
    sae = create_sae(
        d_model=d_model,
        d_sparse=args.d_sparse,
        tied_weights=False,
        lambda_sparsity=args.lambda_sparsity,
    )
    sae = sae.to(device)
    
    print(f"SAE architecture:")
    print(f"  Input: {d_model}")
    print(f"  Sparse features: {args.d_sparse}")
    print(f"  Expansion ratio: {args.d_sparse / d_model:.1f}x")
    print()
    
    # Setup optimizer
    # Use 8-bit Adam for large SAEs to reduce memory (4x less than fp32 Adam)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(sae.parameters(), lr=args.lr)
        print("Using 8-bit Adam optimizer (bitsandbytes)")
    except ImportError:
        print("Warning: bitsandbytes not available, using standard Adam")
        print("For large SAEs, install: pip install bitsandbytes")
        optimizer = optim.Adam(sae.parameters(), lr=args.lr)
    
    # Train SAE
    print("Starting training...")
    print()
    losses = train_sae(
        train_activations,
        sae,
        optimizer,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        normalize_every=100,
        verbose=True,
    )
    
    print()
    print("Training complete!")
    print()
    
    # Compute final sparsity stats
    print("Computing final sparsity statistics...")
    sae.eval()
    with torch.no_grad():
        # Sample 10k examples for stats
        sample_size = min(10000, n_train)
        sample_indices = np.random.choice(n_train, sample_size, replace=False)
        sample_activations = train_activations[sample_indices]
        
        # Normalize (using saved normalization stats)
        act_mean = losses['normalization']['mean']
        act_std = losses['normalization']['std']
        sample_normalized = (sample_activations - act_mean) / act_std
        
        # Compute sparsity stats in batches to avoid OOM for large SAEs
        batch_size_stats = 1000  # Process 1k samples at a time
        all_f = []  # Store all feature activations for multi-threshold analysis
        all_l1 = []
        
        for i in range(0, sample_size, batch_size_stats):
            j = min(i + batch_size_stats, sample_size)
            batch_tensor = torch.from_numpy(sample_normalized[i:j]).float().to(device)
            
            with torch.no_grad():
                f = sae.encode(batch_tensor)
                
                # Accumulate stats
                all_f.append(f.cpu().numpy())
                all_l1.append(f.sum(dim=1).cpu().numpy())
            
            del batch_tensor, f
            torch.cuda.empty_cache()
        
        # Aggregate stats
        all_f = np.concatenate(all_f)  # (sample_size, d_sparse)
        all_l1 = np.concatenate(all_l1)
        
        # Compute L0 norm at multiple thresholds
        thresholds = [1e-5, 1e-3, 1e-2, 5e-2, 0.1]
        l0_by_threshold = {}
        dead_by_threshold = {}
        
        for thresh in thresholds:
            active = (np.abs(all_f) > thresh).astype(float)
            l0_norm = active.sum(axis=1).mean()
            dead_features = (active.sum(axis=0) == 0).sum()
            l0_by_threshold[thresh] = float(l0_norm)
            dead_by_threshold[thresh] = int(dead_features)
        
        l1_norm = all_l1.mean()
        
        sparsity_stats = {
            'l0_by_threshold': l0_by_threshold,
            'dead_by_threshold': dead_by_threshold,
            'l1_norm': float(l1_norm),
        }
    
    print(f"Sparsity statistics (on {sample_size} examples):")
    print(f"  L1 norm: {sparsity_stats['l1_norm']:.2f}")
    print()
    print("  L0 norm by activation threshold:")
    for thresh in thresholds:
        l0 = sparsity_stats['l0_by_threshold'][thresh]
        dead = sparsity_stats['dead_by_threshold'][thresh]
        pct_active = 100.0 * l0 / sae.d_sparse
        pct_dead = 100.0 * dead / sae.d_sparse
        print(f"    Threshold {thresh:6.0e}: L0 = {l0:7.2f} ({pct_active:5.2f}% active), Dead = {dead:5d} ({pct_dead:5.2f}%)")
    print()
    
    # Save SAE
    model_path = out_dir / f"sae_layer{args.layer}_unified.pt"
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {
            'd_model': d_model,
            'd_sparse': args.d_sparse,
            'tied_weights': False,
            'lambda_sparsity': args.lambda_sparsity,
        },
        'training': {
            'lr': args.lr,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'n_train': n_train,
        },
        'normalization': {
            'mean': losses['normalization']['mean'],
            'std': losses['normalization']['std'],
        },
        'final_sparsity': sparsity_stats,
        'loss_history': {
            'total': losses['total'],
            'reconstruction': losses['reconstruction'],
            'sparsity': losses['sparsity'],
        },
    }, model_path)
    
    print(f"Saved SAE to: {model_path}")
    
    # Save training summary
    summary = {
        'layer': args.layer,
        'd_model': d_model,
        'd_sparse': args.d_sparse,
        'n_train': int(n_train),
        'config': {
            'lambda_sparsity': args.lambda_sparsity,
            'lr': args.lr,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
        },
        'final_loss': float(losses['total'][-1]),
        'sparsity_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                          for k, v in sparsity_stats.items()},
    }
    
    summary_path = out_dir / f"training_summary_layer{args.layer}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to: {summary_path}")
    print()
    print("="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
