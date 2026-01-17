#!/usr/bin/env python3
"""Test model prediction with different image sizes."""
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path

# Load checkpoint
checkpoint_path = Path('Model/best_model (1).pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"Val MAE from training: {checkpoint['val_mae']:.4f} g/m²")
print(f"Trained for {checkpoint['epoch']} epochs")

# Define model
class SimpleCNNGSMPredictor(nn.Module):
    def __init__(self):
        super(SimpleCNNGSMPredictor, self).__init__()
        efficientnet = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(1536, 1)
        )
    
    def forward(self, images):
        cnn_out = self.backbone(images)
        cnn_out = torch.flatten(cnn_out, 1)
        output = self.head(cnn_out)
        return output.squeeze()

# Load model
model = SimpleCNNGSMPredictor()
model.load_state_dict(checkpoint['model_state'], strict=False)
model.eval()
print("\n[OK] Model loaded successfully")

# Find test image
test_images = list(Path('split_feature_dataset/test/images').glob('*.jpg'))
if not test_images:
    print("[ERROR] No test images found")
    exit(1)

# Get actual GSM values
import pandas as pd
df_test = pd.read_csv('split_feature_dataset/test/dataset_test.csv')
gsm_dict = dict(zip(df_test['image_name'], df_test['gsm']))

# Test different image sizes
test_sizes = [224, 256, 288, 320, 352]

for test_image_path in test_images[:3]:  # Test first 3 images
    image_name = test_image_path.name
    actual_gsm = gsm_dict.get(image_name, 'UNKNOWN')
    
    print(f"\nTesting image: {image_name}, Actual GSM: {actual_gsm}")
    
    # Load image
    img = cv2.imread(str(test_image_path))
    if img is None:
        print(f"  [ERROR] Failed to load image")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  Original size: {img_rgb.shape[:2]}")
    
    for size in test_sizes:
        img_resized = cv2.resize(img_rgb, (size, size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        with torch.no_grad():
            pred = model(img_tensor).item()
        
        error = abs(pred - actual_gsm) if actual_gsm != 'UNKNOWN' else 'N/A'
        print(f"  Size {size}: Prediction = {pred:.2f}, Error = {error}")
