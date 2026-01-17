#!/usr/bin/env python3
"""Check if there's a scaling factor in the checkpoint."""
import torch
from pathlib import Path

checkpoint = torch.load('Model/best_model (1).pt', map_location='cpu')

print("Checkpoint keys:", checkpoint.keys())
print("Val MAE:", checkpoint.get('val_mae'))
print("Epoch:", checkpoint.get('epoch'))

# Look for any scaling factors
print("\nSearching for scaling/normalization info...")
for key in checkpoint.keys():
    if 'scale' in key.lower() or 'norm' in key.lower() or 'mean' in key.lower():
        print(f"Found: {key} = {checkpoint[key]}")

# Check if there's denormalization info
print("\nFull checkpoint keys:")
for k in checkpoint.keys():
    print(f"  {k}: {type(checkpoint[k])}")
