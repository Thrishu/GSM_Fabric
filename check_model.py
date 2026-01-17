#!/usr/bin/env python3
"""Check model checkpoint details."""
import torch
from pathlib import Path

model_path = Path('Model/best_model (1).pt')
checkpoint = torch.load(model_path, map_location='cpu')

print("Checkpoint info:")
print(f"  Val MAE: {checkpoint.get('val_mae', 'NOT FOUND')}")
print(f"  Epoch: {checkpoint.get('epoch', 'NOT FOUND')}")

print("\nFirst 15 model state keys:")
for i, k in enumerate(list(checkpoint['model_state'].keys())[:15]):
    shape = checkpoint['model_state'][k].shape
    print(f"  {k}: {shape}")

# Look for linear layer to determine output
print("\nLinear layers (head):")
for k in checkpoint['model_state'].keys():
    if 'head' in k or 'fc' in k:
        shape = checkpoint['model_state'][k].shape
        print(f"  {k}: {shape}")
