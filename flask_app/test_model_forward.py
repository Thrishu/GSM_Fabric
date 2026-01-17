#!/usr/bin/env python
"""Test model loading and forward pass."""

import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
IMG_SIZE = checkpoint.get('img_size', 352)

# Load scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Load test data
test_csv = Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'dataset_test.csv'
df_test = pd.read_csv(test_csv)
meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
feature_cols = [col for col in df_test.columns if col not in meta_cols]

# Get first test sample
first_sample = df_test.iloc[0]
image_name = first_sample['image_name']
true_gsm = first_sample['gsm']

# Load image
image_paths = list((Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'images').glob(f'*{image_name}*'))
img_path = image_paths[0]

# Prepare image tensor
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
img = Image.open(img_path).convert('RGB')
img_tensor = inference_transform(img).unsqueeze(0)

# Prepare features tensor
features_raw = first_sample[feature_cols].values
features_scaled = scaler.transform(features_raw.reshape(1, -1))
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

print(f"Input shapes:")
print(f"  Image: {img_tensor.shape}")
print(f"  Features: {features_tensor.shape}")
print(f"  True GSM: {true_gsm}")

# Define model (MUST MATCH CHECKPOINT EXACTLY)
class HybridGSMPredictor(nn.Module):
    """Hybrid model combining EfficientNet-B3 CNN with engineered fabric features."""
    def __init__(self, num_features, dropout=0.5):
        super(HybridGSMPredictor, self).__init__()
        # Pre-trained EfficientNet-B3 backbone
        efficientnet = models.efficientnet_b3(weights='IMAGENET1K_V1')
        # Remove classifier head and wrap as 'backbone' to match checkpoint
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])
        cnn_feature_size = 1536  # EfficientNet-B3 output
        # Feature processing branch
        self.feature_branch = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        # Fusion and prediction head
        combined_size = cnn_feature_size + 128
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(256, 1)
        )
    
    def forward(self, images, features):
        cnn_out = self.backbone(images)
        cnn_out = torch.flatten(cnn_out, 1)
        feat_out = self.feature_branch(features)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        output = self.fusion(combined)
        return output.squeeze()

# Create model
model = HybridGSMPredictor(num_features=len(feature_cols))

# Load checkpoint with strict=False
print(f"\nLoading model from checkpoint...")
model_state_dict = checkpoint['model_state']
incompatible = model.load_state_dict(model_state_dict, strict=False)
print(f"Missing keys: {len(incompatible.missing_keys)}")
print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

# Set to eval mode
model.eval()

# Make prediction
print(f"\nMaking prediction...")
with torch.no_grad():
    prediction = model(img_tensor, features_tensor).item()

print(f"\nResult:")
print(f"  Predicted GSM: {prediction:.2f}")
print(f"  Actual GSM:    {true_gsm:.2f}")
print(f"  Error:         {prediction - true_gsm:+.2f}")

# Test with a few more samples
print(f"\n--- Testing more samples ---\n")
for i in range(min(5, len(df_test))):
    sample = df_test.iloc[i]
    img_name = sample['image_name']
    true = sample['gsm']
    
    img_paths = list((Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'images').glob(f'*{img_name}*'))
    if img_paths:
        img_path = img_paths[0]
        img = Image.open(img_path).convert('RGB')
        img_t = inference_transform(img).unsqueeze(0)
        
        feat_raw = sample[feature_cols].values
        feat_scaled = scaler.transform(feat_raw.reshape(1, -1))
        feat_t = torch.tensor(feat_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred = model(img_t, feat_t).item()
        
        error = pred - true
        print(f"{i+1}. {img_name:20} | True: {true:6.1f} | Pred: {pred:8.2f} | Error: {error:+7.2f}")
    else:
        print(f"{i+1}. {img_name:20} | Image not found!")

print("\n✅ Test complete")
