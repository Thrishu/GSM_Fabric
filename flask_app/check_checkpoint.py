#!/usr/bin/env python
"""Check checkpoint contents."""

import torch
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
print("All checkpoint keys:")
for key in checkpoint.keys():
    val = checkpoint[key]
    if isinstance(val, torch.Tensor):
        print(f"  {key}: Tensor {val.shape}")
    elif isinstance(val, dict):
        subkeys = list(val.keys())[:3]
        print(f"  {key}: Dict with {len(val)} items (sample: {subkeys})")
    else:
        print(f"  {key}: {type(val).__name__}")
