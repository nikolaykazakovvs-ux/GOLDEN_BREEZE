"""Quick test for strategy signals integration."""

import torch
from aimodule.training.v4_dataset import FusionDatasetV2
from aimodule.models.v4_transformer.model import GoldenBreezeFusionV4
from aimodule.models.v4_transformer.config import V4Config
import pandas as pd

print("Loading data...")
df_m5 = pd.read_csv('data/raw/XAUUSD/M5.csv')
df_m5['time'] = pd.to_datetime(df_m5['time'])
df_h1 = pd.read_csv('data/raw/XAUUSD/H1.csv')
df_h1['time'] = pd.to_datetime(df_h1['time'])
df_labels = pd.read_csv('data/labels/direction_labels_XAUUSD_6m.csv')
df_labels['time'] = pd.to_datetime(df_labels['time'])

# Create dataset with small subset for speed
print("Creating dataset...")
config = V4Config()
dataset = FusionDatasetV2.from_dataframes(
    df_m5[:10000], 
    df_h1[:2000], 
    df_labels[:10000], 
    config,
    label_col='direction_label',
)
print(f'Dataset size: {len(dataset)}')

# Get first sample
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')

if 'strategy_signals' in sample:
    print(f'strategy_signals shape: {sample["strategy_signals"].shape}')
    print(f'strategy_signals sample: {sample["strategy_signals"][:5]}')
else:
    print("WARNING: strategy_signals NOT FOUND in sample!")

# Create model
print("\nCreating model...")
model = GoldenBreezeFusionV4(config).cuda()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print("\nTesting forward pass...")
# Convert keys to model expected format
batch = {
    'x_fast_ohlcv': sample['x_fast'].unsqueeze(0).cuda(),
    'x_slow_ohlcv': sample['x_slow'].unsqueeze(0).cuda(),
    'smc_static': sample['smc_static'].unsqueeze(0).cuda(),
    'smc_dynamic': sample['smc_dynamic'].unsqueeze(0).cuda(),
    'strategy_signals': sample['strategy_signals'].unsqueeze(0).cuda(),
}
print(f"Batch keys: {list(batch.keys())}")

try:
    output = model(**batch)
    print(f'Output class_logits shape: {output["class_logits"].shape}')
    print(f'Predicted class: {output["predicted_class"].item()}')
    print('\n✅ Forward pass SUCCESS!')
except Exception as e:
    print(f'\n❌ Forward pass FAILED: {e}')
    import traceback
    traceback.print_exc()
