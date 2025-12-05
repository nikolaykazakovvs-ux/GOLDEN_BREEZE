"""
Export Scaler Parameters for v5 Ultimate Inference

Loads the training dataset and computes mean/std for normalization.
These values are essential for live inference - we must normalize 
live data exactly the same way as training data.

Author: Golden Breeze Team
Version: 1.0.0
Date: 2025-12-05
"""

import json
import numpy as np
from pathlib import Path


def export_scalers(
    dataset_path: str = "data/prepared/v4_6year_dataset.npz",
    output_path: str = "models/v5_ultimate/scaler_params.json"
):
    """
    Calculate and export normalization parameters from training data.
    
    For each input type (x_fast, x_slow, x_strategy), we compute:
    - mean: per-feature mean across all samples and timesteps
    - std: per-feature std across all samples and timesteps
    
    Args:
        dataset_path: Path to training .npz file
        output_path: Path to save JSON with scaler params
    """
    print("=" * 60)
    print("📊 Exporting Scaler Parameters for v5 Ultimate")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {dataset_path}")
    data = np.load(dataset_path)
    
    x_fast = data['x_fast']      # (N, 50, 15) - M5 bars
    x_slow = data['x_slow']      # (N, 20, 8)  - H1 bars
    x_strategy = data['x_strategy']  # (N, 64) - strategy features
    
    print(f"   x_fast: {x_fast.shape}")
    print(f"   x_slow: {x_slow.shape}")
    print(f"   x_strategy: {x_strategy.shape}")
    
    # Calculate statistics for x_fast (M5 features)
    # Shape: (N, 50, 15) -> compute per-feature (axis 0,1)
    print("\n📈 Computing statistics for x_fast (M5 features)...")
    fast_mean = np.mean(x_fast, axis=(0, 1))  # (15,)
    fast_std = np.std(x_fast, axis=(0, 1))    # (15,)
    fast_std = np.where(fast_std < 1e-8, 1.0, fast_std)  # Avoid division by zero
    
    print(f"   fast_mean shape: {fast_mean.shape}")
    print(f"   fast_std shape: {fast_std.shape}")
    print(f"   Sample means: {fast_mean[:5].round(4)}")
    
    # Calculate statistics for x_slow (H1 features)
    # Shape: (N, 20, 8) -> compute per-feature (axis 0,1)
    print("\n📈 Computing statistics for x_slow (H1 features)...")
    slow_mean = np.mean(x_slow, axis=(0, 1))  # (8,)
    slow_std = np.std(x_slow, axis=(0, 1))    # (8,)
    slow_std = np.where(slow_std < 1e-8, 1.0, slow_std)
    
    print(f"   slow_mean shape: {slow_mean.shape}")
    print(f"   slow_std shape: {slow_std.shape}")
    print(f"   Sample means: {slow_mean[:5].round(4)}")
    
    # Calculate statistics for x_strategy (strategy features)
    # Shape: (N, 64) -> compute per-feature (axis 0)
    print("\n📈 Computing statistics for x_strategy (strategy features)...")
    strat_mean = np.mean(x_strategy, axis=0)  # (64,)
    strat_std = np.std(x_strategy, axis=0)    # (64,)
    strat_std = np.where(strat_std < 1e-8, 1.0, strat_std)
    
    print(f"   strat_mean shape: {strat_mean.shape}")
    print(f"   strat_std shape: {strat_std.shape}")
    print(f"   Sample means: {strat_mean[:5].round(4)}")
    
    # Build output dictionary
    scaler_params = {
        # M5 features (15 features)
        'fast_mean': fast_mean.tolist(),
        'fast_std': fast_std.tolist(),
        
        # H1 features (8 features)
        'slow_mean': slow_mean.tolist(),
        'slow_std': slow_std.tolist(),
        
        # Strategy features (64 features)
        'strat_mean': strat_mean.tolist(),
        'strat_std': strat_std.tolist(),
        
        # Metadata
        'dataset_samples': int(x_fast.shape[0]),
        'fast_seq_len': int(x_fast.shape[1]),
        'fast_features': int(x_fast.shape[2]),
        'slow_seq_len': int(x_slow.shape[1]),
        'slow_features': int(x_slow.shape[2]),
        'strat_features': int(x_strategy.shape[1]),
        'version': '5.0.0-ultimate',
    }
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    print(f"\n💾 Saving scaler parameters to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    print("\n✅ Scaler parameters exported successfully!")
    print(f"   Total features: {15 + 8 + 64} = 87")
    print(f"   Samples used: {x_fast.shape[0]:,}")
    
    return scaler_params


def verify_scalers(scaler_path: str = "models/v5_ultimate/scaler_params.json"):
    """Verify that scaler params can be loaded correctly."""
    print("\n🔍 Verifying scaler parameters...")
    
    with open(scaler_path, 'r') as f:
        params = json.load(f)
    
    print(f"   fast_mean: {len(params['fast_mean'])} values")
    print(f"   fast_std: {len(params['fast_std'])} values")
    print(f"   slow_mean: {len(params['slow_mean'])} values")
    print(f"   slow_std: {len(params['slow_std'])} values")
    print(f"   strat_mean: {len(params['strat_mean'])} values")
    print(f"   strat_std: {len(params['strat_std'])} values")
    print(f"   Version: {params['version']}")
    print("\n✅ Verification passed!")
    
    return params


if __name__ == "__main__":
    # Export scalers
    params = export_scalers()
    
    # Verify
    verify_scalers()
