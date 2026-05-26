"""
MLP Configuration for DeepSeek 7B/8B Calibration Probes

Architecture designed for predicting calibration residuals from hidden states.
Optimized for DeepSeek models with hidden dimension 4096.
"""

# MLP Architecture
MLP_CONFIG = {
    # Input/Output dimensions
    "input_dim": 4096,  # DeepSeek hidden dimension
    "output_dim": 1,    # Single residual value per option
    
    # Hidden layers configuration
    "hidden_dims": [1024, 512, 256, 128],  # Four hidden layers
    
    # Activation function
    "activation": "relu",  # Options: 'relu', 'gelu', 'tanh'
    "output_activation": "tanh",  # Output activation: 'tanh' ([-1,1]), 'sigmoid' ([0,1]), None (unbounded)
    "output_scale": 1.0,  # Scale factor for output (tanh gives [-1,1], scale by 1.0 keeps [-1,1])
    
    # Regularization
    "dropout": 0.2,  # Dropout rate (0.0 = no dropout)
    "weight_decay_grid": [0, 0.1, 1.0, 5, 10.0, 25, 30, 45.0],  # Reduced grid for faster training (4 values)
    "output_penalty_grid": [0, 0.01, 0.1, 0.25, 0.5],  # Reduced grid (3 values) → 12 total combinations per layer
    
    # Training hyperparameters
    "learning_rate": 1e-3,
    "batch_size": 256,
    "num_epochs": 100,
    "patience": 30,  # Early stopping patience (increased for longer training)
    
    # Optimization
    "optimizer": "adamw",  # Options: 'adam', 'adamw', 'sgd'
    "lr_scheduler": "reduce_on_plateau",  # Options: 'reduce_on_plateau', 'cosine', None
    "lr_patience": 5,  # For ReduceLROnPlateau
    "lr_factor": 0.5,  # LR reduction factor
    
    # Normalization
    "batch_norm": False,  # Use batch normalization
    "layer_norm": False,  # Use layer normalization
    
    # Data preprocessing
    "normalize_input": True,  # Z-score normalization
    
    # Loss function
    "loss_fn": "mse",  # Options: 'mse', 'huber', 'mae'
    
    # Validation
    "val_split": 0.2,  # Fraction for validation during training
    "shuffle": True,
    
    # Misc
    "seed": 42,
    "verbose": True,
}

# Alternative configurations for experimentation

# Deeper network (better capacity, slower)
MLP_CONFIG_DEEP = {
    **MLP_CONFIG,
    "hidden_dims": [2048, 1024, 512, 256],
    "dropout": 0.3,
    "num_epochs": 150,
}

# Shallower network (faster, may underfit)
MLP_CONFIG_SHALLOW = {
    **MLP_CONFIG,
    "hidden_dims": [512],
    "dropout": 0.1,
    "num_epochs": 50,
}

# Wide network (more capacity per layer)
MLP_CONFIG_WIDE = {
    **MLP_CONFIG,
    "hidden_dims": [2048, 1024],
    "dropout": 0.25,
    "batch_size": 128,
}
