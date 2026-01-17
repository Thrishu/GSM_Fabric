#!/usr/bin/env python
"""Test script to debug GSM prediction."""

import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.models as models

# Paths
MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'

# Load checkpoint
print("Loading checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
print(f"Checkpoint keys: {checkpoint.keys()}")

# Check if model_state exists
if 'model_state' in checkpoint:
    model_state_dict = checkpoint['model_state']
    print(f"Found model_state with keys: {list(model_state_dict.keys())[:5]}...")
    
    # Find the feature branch to understand model structure
    for key in model_state_dict.keys():
        if 'feature_branch' in key:
            print(f"Feature branch key: {key}, shape: {model_state_dict[key].shape}")
            break

# Load scaler
print("\nLoading scaler...")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
print(f"Scaler type: {type(scaler)}")
print(f"Scaler center (mean): {scaler.center_}")
print(f"Scaler scale: {scaler.scale_}")

# Load test data
print("\nLoading test data...")
test_csv = Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'dataset_test.csv'
if test_csv.exists():
    df_test = pd.read_csv(test_csv)
    meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
    feature_cols = [col for col in df_test.columns if col not in meta_cols]
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns sample: {feature_cols[:5]}")
    
    # Get first test sample
    first_sample = df_test.iloc[0]
    true_gsm = first_sample['gsm']
    print(f"\nFirst sample - True GSM: {true_gsm}")
    
    # Extract features
    features = first_sample[feature_cols].values.reshape(1, -1)
    print(f"Raw features shape: {features.shape}")
    print(f"Raw features (first 5): {features[0, :5]}")
    
    # Scale features
    features_scaled = scaler.transform(features)
    print(f"Scaled features (first 5): {features_scaled[0, :5]}")
    
    # Test with different inputs
    print("\n--- Testing prediction with different inputs ---")
    
    # Test 1: Dummy image + scaled features
    print("\nTest 1: Dummy zero image + scaled features")
    dummy_image = torch.zeros((1, 3, 224, 224))
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    
    print(f"Dummy image shape: {dummy_image.shape}")
    print(f"Features tensor shape: {features_tensor.shape}")
    
    # Test 2: Check for outliers in features
    print(f"\nFeature statistics:")
    print(f"Min: {features[0].min()}, Max: {features[0].max()}, Mean: {features[0].mean()}")
    print(f"Scaled Min: {features_scaled[0].min()}, Scaled Max: {features_scaled[0].max()}, Scaled Mean: {features_scaled[0].mean()}")
