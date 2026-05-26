"""
Sparse Autoencoder (SAE) for analyzing calibration-binned activations.

This SAE will be trained separately for each confidence bin to discover
bin-specific features in the model's internal representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationBinSAE(nn.Module):
    """
    Sparse Autoencoder for calibration circuit discovery
    
    Architecture:
    x (d_model) → Encoder → f (d_sparse) → Decoder → x̂ (d_model)
    
    Uses ReLU activation for non-negativity and natural sparsity.
    Includes decoder weight normalization for training stability.
    """
    
    def __init__(
        self,
        d_model: int = 4096,        # DeepSeek hidden dimension
        d_sparse: int = 16384,      # Sparse feature dimension (4x expansion)
        tied_weights: bool = False, # Whether to tie encoder/decoder weights
        lambda_sparsity: float = 0.01,  # Sparsity penalty weight
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_sparse = d_sparse
        self.tied_weights = tied_weights
        self.lambda_sparsity = lambda_sparsity
        
        # ===== Core Autoencoder Components =====
        
        # Encoder: d_model → d_sparse
        self.encoder = nn.Linear(d_model, d_sparse, bias=True)
        
        # Decoder: d_sparse → d_model
        if tied_weights:
            # Decoder weights = Encoder weights transposed
            # Only learn encoder, decoder is derived
            self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        else:
            # Independent decoder weights (more common for SAEs)
            self.decoder = nn.Linear(d_sparse, d_model, bias=True)
        
        # ===== Initialization =====
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights following best practices for SAEs"""
        
        # Encoder: Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Decoder: Xavier/Glorot initialization
        if not self.tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
        else:
            nn.init.zeros_(self.decoder_bias)
        
        # Normalize decoder columns to unit norm (important for SAEs!)
        self._normalize_decoder()
    
    def _normalize_decoder(self):
        """Normalize decoder columns to unit norm (critical for SAE training!)"""
        if self.tied_weights:
            # Normalize encoder rows (which become decoder columns when transposed)
            with torch.no_grad():
                norms = torch.norm(self.encoder.weight, dim=0, keepdim=True)
                self.encoder.weight.div_(norms + 1e-8)
        else:
            # Normalize decoder columns directly
            with torch.no_grad():
                norms = torch.norm(self.decoder.weight, dim=0, keepdim=True)
                self.decoder.weight.div_(norms + 1e-8)
    
    def encode(self, x):
        """
        Encode activations to sparse features
        
        Args:
            x: (batch_size, d_model) activation vectors
        
        Returns:
            f: (batch_size, d_sparse) sparse feature activations
        """
        # Linear projection
        pre_activation = self.encoder(x)
        
        # ReLU for non-negativity and sparsity
        f = F.relu(pre_activation)
        
        return f
    
    def decode(self, f):
        """
        Decode sparse features back to activation space
        
        Args:
            f: (batch_size, d_sparse) sparse feature activations
        
        Returns:
            x_hat: (batch_size, d_model) reconstructed activations
        """
        if self.tied_weights:
            # Decoder = Encoder^T
            x_hat = F.linear(f, self.encoder.weight.t(), self.decoder_bias)
        else:
            x_hat = self.decoder(f)
        
        return x_hat
    
    def forward(self, x, compute_loss=False):
        """
        Full forward pass
        
        Args:
            x: (batch_size, d_model) input activations
            compute_loss: whether to compute and return loss
        
        Returns:
            If compute_loss: (x_hat, f, loss_dict)
            Otherwise: (x_hat, f)
        """
        # Encode
        f = self.encode(x)
        
        # Decode
        x_hat = self.decode(f)
        
        if compute_loss:
            loss_dict = self.compute_loss(x, x_hat, f)
            return x_hat, f, loss_dict
        
        return x_hat, f
    
    def compute_loss(self, x, x_hat, f):
        """
        Compute total loss: reconstruction + sparsity
        
        Loss components:
        1. Reconstruction: MSE(x, x_hat)
        2. Sparsity: weighted L1 on features
        
        Returns:
            dict with 'total', 'reconstruction', 'sparsity' losses
        """
        # ===== 1. Reconstruction Loss =====
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
        
        # ===== 2. Sparsity Loss =====
        # L1 penalty weighted by decoder norm
        # This encourages features to be sparse AND decorrelated
        
        if self.tied_weights:
            decoder_norms = torch.norm(self.encoder.weight, dim=0)
        else:
            decoder_norms = torch.norm(self.decoder.weight, dim=0)
        
        # Sparsity: sum over features, mean over batch
        sparsity_loss = (f * decoder_norms.unsqueeze(0)).sum(dim=1).mean()
        
        # ===== Total Loss =====
        total_loss = reconstruction_loss + self.lambda_sparsity * sparsity_loss
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'sparsity': sparsity_loss,
        }
    
    def get_feature_activations(self, x):
        """Convenience method to just get feature activations"""
        with torch.no_grad():
            return self.encode(x)
    
    def reconstruct(self, x):
        """Convenience method to get reconstruction"""
        with torch.no_grad():
            f = self.encode(x)
            return self.decode(f)
    
    @torch.no_grad()
    def normalize_decoder_weights(self):
        """
        Normalize decoder weights (should be called periodically during training)
        This is important for stable SAE training!
        """
        self._normalize_decoder()
    
    @torch.no_grad()
    def get_sparsity_stats(self, x, threshold=1e-5):
        """
        Compute sparsity statistics for given inputs
        
        Args:
            x: (batch_size, d_model) input activations
            threshold: threshold below which a feature is considered "inactive"
        
        Returns:
            dict with sparsity metrics
        """
        f = self.encode(x)
        
        # Features active above threshold
        active = (f > threshold).float()
        
        # Statistics
        l0_norm = active.sum(dim=1).mean().item()  # Avg active features per example
        l1_norm = f.sum(dim=1).mean().item()       # Avg L1 norm
        
        # Per-feature statistics
        feature_activation_freq = active.mean(dim=0)  # How often each feature activates
        dead_features = (feature_activation_freq == 0).sum().item()
        
        return {
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'pct_active': 100 * l0_norm / self.d_sparse,
            'dead_features': dead_features,
            'pct_dead': 100 * dead_features / self.d_sparse,
        }


def create_sae(
    d_model: int = 4096,
    d_sparse: int = 16384,
    tied_weights: bool = False,
    lambda_sparsity: float = 0.01,
):
    """
    Factory function to create a SAE model
    
    Args:
        d_model: Hidden dimension of the model
        d_sparse: Sparse feature dimension (typically 4x d_model)
        tied_weights: Whether to tie encoder/decoder weights
        lambda_sparsity: Sparsity penalty weight
    
    Returns:
        CalibrationBinSAE model
    """
    return CalibrationBinSAE(
        d_model=d_model,
        d_sparse=d_sparse,
        tied_weights=tied_weights,
        lambda_sparsity=lambda_sparsity,
    )
