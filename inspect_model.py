
import torch
import sys

path = "Model/gsm_regressor.pt"
try:
    print(f"Loading {path}...")
    checkpoint = torch.load(path, map_location='cpu')
    print(f"Type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Keys: {list(checkpoint.keys())}")
        for k, v in checkpoint.items():
            if hasattr(v, 'keys'):
                print(f"  Key '{k}' is dict with keys: {list(v.keys())[:5]}...")
            else:
                print(f"  Key '{k}' type: {type(v)}")
    else:
        print("Not a dictionary.")
except Exception as e:
    print(f"Error: {e}")
