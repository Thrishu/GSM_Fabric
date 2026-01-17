#!/usr/bin/env python
"""Test prediction with actual image from dataset."""

import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
print(f"img_size from checkpoint: {checkpoint['img_size']}")

# Load scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Load test data
test_csv = Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'dataset_test.csv'
df_test = pd.read_csv(test_csv)
meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
feature_cols = [col for col in df_test.columns if col not in meta_cols]

# Get first sample
first_sample = df_test.iloc[0]
image_name = first_sample['image_name']
true_gsm = first_sample['gsm']

print(f"\nTest sample:")
print(f"  Image: {image_name}")
print(f"  True GSM: {true_gsm}")

# Load the actual image
image_paths = list((Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'images').glob(f'*{image_name}*'))
if image_paths:
    img_path = image_paths[0]
    print(f"  Image path: {img_path}")
    
    # Load and preprocess image
    img = cv2.imread(str(img_path))
    if img is not None:
        print(f"  Original image shape: {img.shape}")
        
        # Resize to checkpoint size
        img_resized = cv2.resize(img, (checkpoint['img_size'], checkpoint['img_size']))
        print(f"  Resized image shape: {img_resized.shape}")
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        print(f"  Tensor shape: {img_tensor.shape}")
        
        # Get features from dataset
        features = first_sample[feature_cols].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        print(f"\n  Features tensor shape: {features_tensor.shape}")
        print(f"  First 5 raw features: {features[:, :5]}")
        print(f"  First 5 scaled features: {features_scaled[0, :5]}")
        
        # Test 1: With real image
        print(f"\n--- Test 1: Real image + scaled features ---")
        print(f"  Raw image tensor stats: min={img_tensor.min():.3f}, max={img_tensor.max():.3f}, mean={img_tensor.mean():.3f}")
        
        # Test 2: With zero image
        print(f"\n--- Test 2: Zero image + scaled features ---")
        zero_image = torch.zeros_like(img_tensor)
        print(f"  Zero image tensor stats: min={zero_image.min():.3f}, max={zero_image.max():.3f}, mean={zero_image.mean():.3f}")
    else:
        print(f"  Could not load image from {img_path}")
else:
    print(f"  No image found matching {image_name}")
    # List available images
    img_dir = Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'images'
    if img_dir.exists():
        available = list(img_dir.glob('*'))[:3]
        print(f"  Available images sample: {[p.name for p in available]}")
