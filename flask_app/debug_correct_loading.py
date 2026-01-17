#!/usr/bin/env python
"""Debug the exact prediction with proper logging."""

import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
IMG_SIZE = checkpoint.get('img_size', 352)
print(f"Model expects image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Checkpoint keys: {checkpoint.keys()}")

# Load scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print(f"\nScaler info:")
print(f"  Feature count: {scaler.n_features_in_}")
print(f"  Center shape: {scaler.center_.shape}")

# Load test data
test_csv = Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'dataset_test.csv'
df_test = pd.read_csv(test_csv)
meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
feature_cols = [col for col in df_test.columns if col not in meta_cols]

print(f"\nFeature columns count: {len(feature_cols)}")

# Get first test sample
first_sample = df_test.iloc[0]
image_name = first_sample['image_name']
true_gsm = first_sample['gsm']

print(f"\nTest sample:")
print(f"  Image: {image_name}")
print(f"  True GSM: {true_gsm}")

# Find and load image
image_paths = list((Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'images').glob(f'*{image_name}*'))
if image_paths:
    img_path = image_paths[0]
    print(f"  Image path: {img_path}")
    
    # Load image using PIL (like the notebook)
    img = Image.open(img_path).convert('RGB')
    print(f"  PIL Image size: {img.size}")
    
    # Apply the EXACT transforms from the notebook (without augmentation for inference)
    inference_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = inference_transform(img).unsqueeze(0)
    print(f"\n  Image tensor shape: {img_tensor.shape}")
    print(f"  Image tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
    
    # Get features - EXACTLY as in notebook
    features_raw = first_sample[feature_cols].values
    print(f"\n  Raw features shape: {features_raw.shape}")
    print(f"  Raw features sample (first 5): {features_raw[:5]}")
    
    # Scale features with scaler
    features_scaled = scaler.transform(features_raw.reshape(1, -1))
    print(f"  Scaled features sample (first 5): {features_scaled[0, :5]}")
    
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    print(f"  Features tensor shape: {features_tensor.shape}")
    
    # Now let's test different image sizes to see which works
    print(f"\n--- Testing different image preprocessing ---")
    
    for test_size in [224, 352]:
        test_transform = transforms.Compose([
            transforms.Resize((test_size, test_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_test = Image.open(img_path).convert('RGB')
        img_test_tensor = test_transform(img_test).unsqueeze(0)
        
        print(f"\nTest size {test_size}x{test_size}:")
        print(f"  Tensor shape: {img_test_tensor.shape}")
        print(f"  Tensor range: [{img_test_tensor.min():.4f}, {img_test_tensor.max():.4f}]")
        print(f"  Tensor mean: {img_test_tensor.mean():.4f}")

else:
    print(f"ERROR: No image found matching {image_name}")
