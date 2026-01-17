#!/usr/bin/env python
"""Deep inspection of checkpoint."""

import torch
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'

print("=" * 60)
print("CHECKPOINT INSPECTION")
print("=" * 60)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# Check top-level keys
print("\nTop-level keys in checkpoint:")
for key in checkpoint.keys():
    val = checkpoint[key]
    if isinstance(val, torch.Tensor):
        print(f"  {key}: Tensor shape {val.shape}, dtype {val.dtype}")
    elif isinstance(val, dict):
        print(f"  {key}: Dict with {len(val)} items")
    elif isinstance(val, list):
        print(f"  {key}: List with {len(val)} items")
    else:
        print(f"  {key}: {type(val).__name__}")
        if isinstance(val, (int, float, str)):
            print(f"      Value: {val}")

# Check model_state keys (first 10)
print("\nFirst 10 keys in model_state:")
if 'model_state' in checkpoint:
    model_state = checkpoint['model_state']
    for i, key in enumerate(list(model_state.keys())[:10]):
        tensor = model_state[key]
        print(f"  {key}: shape {tensor.shape if hasattr(tensor, 'shape') else 'N/A'}")

# Check scaler
print("\n" + "=" * 60)
print("SCALER INSPECTION")
print("=" * 60)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print(f"\nScaler type: {type(scaler)}")
print(f"Scaler attributes:")
for attr in ['center_', 'scale_', 'quantile_', 'n_features_in_']:
    if hasattr(scaler, attr):
        val = getattr(scaler, attr)
        if hasattr(val, 'shape'):
            print(f"  {attr}: shape {val.shape}, dtype {val.dtype if hasattr(val, 'dtype') else 'N/A'}")
        else:
            print(f"  {attr}: {val}")

# Check feature names
print(f"\nScaler feature_names_in_:")
if hasattr(scaler, 'feature_names_in_'):
    print(f"  {scaler.feature_names_in_}")
