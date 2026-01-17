#!/usr/bin/env python3
"""Test Flask app model loading and predictions."""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add flask_app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flask_app'))

# Set up Flask before importing
os.environ['FLASK_ENV'] = 'development'

from app import load_model_and_scaler, predict_gsm, model as global_model

# Test 1: Load model
print("=" * 60)
print("TEST 1: Loading Model")
print("=" * 60)
load_model_and_scaler()

# Need to access the global model from the app module
import app
if app.model is not None:
    print("[OK] Model loaded successfully!")
    print(f"[OK] Model type: {type(app.model)}")
else:
    print("[ERROR] Model failed to load!")
    sys.exit(1)

# Test 2: Find test image
print("\n" + "=" * 60)
print("TEST 2: Finding Test Image")
print("=" * 60)

test_image_dirs = [
    Path(__file__).parent / 'data' / 'augmented_dataset' / 'images',
    Path(__file__).parent / 'data' / 'split_feature_dataset' / 'test' / 'images',
    Path(__file__).parent / 'split_feature_dataset' / 'test' / 'images',
    Path(__file__).parent / 'preprocessed_dataset' / 'images',
]

test_image_path = None
for img_dir in test_image_dirs:
    if img_dir.exists():
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        if images:
            test_image_path = str(images[0])
            print(f"[OK] Found test image: {test_image_path}")
            break

if test_image_path is None:
    print("[ERROR] No test image found!")
    sys.exit(1)

# Test 3: Make prediction
print("\n" + "=" * 60)
print("TEST 3: Making Prediction")
print("=" * 60)

prediction, error = predict_gsm({}, test_image_path)

if error:
    print(f"[ERROR] Prediction failed: {error}")
    sys.exit(1)

print(f"[OK] Prediction successful!")
print(f"[OK] Predicted GSM: {prediction:.2f} g/m²")

# Test 4: Validate prediction range
print("\n" + "=" * 60)
print("TEST 4: Validating Prediction Range")
print("=" * 60)

# Expected range: 85-297 g/m²
if 50 < prediction < 350:
    print(f"[OK] Prediction {prediction:.2f} is in reasonable range (50-350)")
else:
    print(f"[WARNING] Prediction {prediction:.2f} is OUTSIDE expected range (50-350)")

if 85 <= prediction <= 297:
    print(f"[OK] Prediction {prediction:.2f} is in training range (85-297)")
else:
    print(f"[WARNING] Prediction {prediction:.2f} is outside training range (85-297)")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
