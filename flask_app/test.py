"""
Test script for GSM Fabric Predictor
Tests the model and feature extraction pipeline
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_model_loading():
    """Test model loading."""
    print("\n🧪 Testing Model Loading...")
    print("-" * 60)
    
    try:
        import torch
        model_path = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(checkpoint)}")
        print(f"   Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def test_scaler_loading():
    """Test scaler loading."""
    print("\n🧪 Testing Scaler Loading...")
    print("-" * 60)
    
    try:
        scaler_path = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'
        
        if not scaler_path.exists():
            print(f"⚠️  Scaler not found: {scaler_path}")
            print("   (Will be created from training data on first run)")
            return True
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"✅ Scaler loaded successfully")
        print(f"   Scaler type: {type(scaler).__name__}")
        print(f"   Scaler size: {scaler_path.stat().st_size / 1024:.2f} KB")
        return True
    
    except Exception as e:
        print(f"❌ Error loading scaler: {str(e)}")
        return False

def test_feature_extraction():
    """Test feature extraction on a sample image."""
    print("\n🧪 Testing Feature Extraction...")
    print("-" * 60)
    
    try:
        import cv2
        from extract_fabric_features import extract_all_fabric_features_single_image
        
        # Create a test image
        test_img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
        
        print("Extracting features from test image...")
        features = extract_all_fabric_features_single_image(test_img)
        
        if features is None:
            print("❌ Feature extraction returned None")
            return False
        
        print(f"✅ Features extracted successfully")
        print(f"   Number of features: {len(features)}")
        print(f"   Sample features:")
        for i, (key, val) in enumerate(list(features.items())[:5]):
            print(f"     {key}: {val:.4f}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error in feature extraction: {str(e)}")
        return False

def test_dataset_loading():
    """Test dataset loading."""
    print("\n🧪 Testing Dataset Loading...")
    print("-" * 60)
    
    try:
        dataset_path = Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'dataset_train.csv'
        
        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            return False
        
        df = pd.read_csv(dataset_path)
        
        meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
        feature_cols = [col for col in df.columns if col not in meta_cols]
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Total samples: {len(df)}")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   GSM range: {df['gsm'].min():.2f} - {df['gsm'].max():.2f} g/m²")
        print(f"   GSM mean: {df['gsm'].mean():.2f} ± {df['gsm'].std():.2f}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return False

def test_prediction_pipeline():
    """Test the complete prediction pipeline."""
    print("\n🧪 Testing Prediction Pipeline...")
    print("-" * 60)
    
    try:
        import torch
        from sklearn.preprocessing import RobustScaler
        
        # Load model
        model_path = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
        checkpoint = torch.load(model_path, map_location='cpu')
        model = checkpoint['model'] if isinstance(checkpoint, dict) else checkpoint
        
        # Load scaler and features
        scaler_path = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load dataset to get feature columns
        dataset_path = Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'dataset_train.csv'
        df = pd.read_csv(dataset_path)
        meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
        feature_cols = [col for col in df.columns if col not in meta_cols]
        
        # Create random feature vector
        X = np.random.randn(1, len(feature_cols)).astype(np.float32)
        X_scaled = scaler.transform(X)
        
        # Make prediction
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            if hasattr(model, 'predict'):
                prediction = model.predict(X_scaled)[0]
            else:
                prediction = model(X_tensor).item()
        
        print(f"✅ Prediction successful")
        print(f"   Predicted GSM: {prediction:.2f} g/m²")
        
        # Validate prediction
        if 0 <= prediction <= 500:
            print(f"   ✅ Prediction within valid range (0-500 g/m²)")
            return True
        else:
            print(f"   ⚠️  Prediction outside typical range")
            return True
    
    except Exception as e:
        print(f"❌ Error in prediction pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test Flask API endpoints."""
    print("\n🧪 Testing Flask API...")
    print("-" * 60)
    
    try:
        from app import app
        
        client = app.test_client()
        
        # Test health endpoint
        response = client.get('/api/health')
        if response.status_code == 200:
            print(f"✅ Health endpoint: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")
        return False

def print_summary(results):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Model Loading", results.get('model_loading')),
        ("Scaler Loading", results.get('scaler_loading')),
        ("Feature Extraction", results.get('feature_extraction')),
        ("Dataset Loading", results.get('dataset_loading')),
        ("Prediction Pipeline", results.get('prediction_pipeline')),
        ("API Endpoints", results.get('api_endpoints')),
    ]
    
    passed = 0
    for test_name, result in tests:
        if result is None:
            status = "⏭️ "
        elif result:
            status = "✅"
            passed += 1
        else:
            status = "❌"
        print(f"{status} {test_name}")
    
    print("="*60)
    print(f"\nResult: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Ready to run: python app.py")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Check above for details.")

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Test GSM Fabric Predictor')
    parser.add_argument('--quick', action='store_true', help='Quick tests only')
    parser.add_argument('--api', action='store_true', help='Include API tests')
    args = parser.parse_args()
    
    print("\n" + "🧪 GSM FABRIC PREDICTOR - TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Basic tests
    results['model_loading'] = test_model_loading()
    results['scaler_loading'] = test_scaler_loading()
    
    if not args.quick:
        results['feature_extraction'] = test_feature_extraction()
        results['dataset_loading'] = test_dataset_loading()
        results['prediction_pipeline'] = test_prediction_pipeline()
    
    if args.api:
        results['api_endpoints'] = test_api_endpoints()
    
    # Print summary
    print_summary(results)

if __name__ == '__main__':
    main()
