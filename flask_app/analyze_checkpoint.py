#!/usr/bin/env python
"""Analyze the actual checkpoint architecture."""

import torch
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model_state = checkpoint['model_state']

# Get unique prefixes
prefixes = set()
for key in model_state.keys():
    prefix = key.split('.')[0]
    prefixes.add(prefix)

print(f"Model architecture prefixes in checkpoint:")
print(f"  {sorted(prefixes)}")

print(f"\nTotal keys in model_state: {len(model_state)}")

# Show sample keys for each prefix
for prefix in sorted(prefixes):
    keys_with_prefix = [k for k in model_state.keys() if k.startswith(prefix)]
    print(f"\n{prefix}: {len(keys_with_prefix)} keys")
    for key in keys_with_prefix[:3]:
        shape = model_state[key].shape if hasattr(model_state[key], 'shape') else 'N/A'
        print(f"    {key} : {shape}")

# Check if the model was saved as a different architecture
print(f"\n\nLikely architecture:")
if 'backbone' in prefixes and 'feature_branch' in prefixes:
    print("  - Has 'backbone' (CNN)")
    print("  - Has 'feature_branch' (tabular features)")
    print("  - Has 'fusion' (prediction head)")
    print("  => This is a HYBRID model")
    
    # Count backbone keys
    backbone_keys = [k for k in model_state.keys() if k.startswith('backbone')]
    print(f"\n  Backbone has {len(backbone_keys)} keys")
    
    feature_keys = [k for k in model_state.keys() if k.startswith('feature_branch')]
    print(f"  Feature branch has {len(feature_keys)} keys")
    
    fusion_keys = [k for k in model_state.keys() if k.startswith('fusion')]
    print(f"  Fusion has {len(fusion_keys)} keys")
