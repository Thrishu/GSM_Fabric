#!/usr/bin/env python3
"""
Inspect what's in the feature_scaler.pkl file
"""
import pickle
import sys
from pathlib import Path

scaler_path = Path(__file__).parent.parent / 'Model' / 'feature_scaler.pkl'

if not scaler_path.exists():
    print(f"Scaler not found: {scaler_path}")
    sys.exit(1)

print(f"Loading from: {scaler_path}\n")

with open(scaler_path, 'rb') as f:
    data = pickle.load(f)

print(f"Type of data: {type(data)}")
print(f"\nKeys (if dict): {data.keys() if isinstance(data, dict) else 'Not a dict'}")

if isinstance(data, dict):
    for key in data.keys():
        value = data[key]
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
            print(f"  First 10: {value[:10]}")
        elif hasattr(value, 'n_features_in_'):
            print(f"  n_features_in_: {value.n_features_in_}")
else:
    # It's just the scaler
    print(f"\nScaler attributes:")
    if hasattr(data, 'n_features_in_'):
        print(f"  n_features_in_: {data.n_features_in_}")
    if hasattr(data, 'feature_names_in_'):
        print(f"  feature_names_in_: {data.feature_names_in_}")
    if hasattr(data, 'scale_'):
        print(f"  scale_ shape: {data.scale_.shape}")
