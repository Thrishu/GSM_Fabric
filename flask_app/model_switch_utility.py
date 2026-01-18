#!/usr/bin/env python3
"""
Model Switching Utility - Test and switch between PyTorch and CatBoost models
"""

import requests
import json
import sys
from pathlib import Path

BASE_URL = 'http://localhost:5000'

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_health():
    """Check app health and current model"""
    try:
        response = requests.get(f'{BASE_URL}/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_header("HEALTH CHECK")
            print(f"Status: {data['status']}")
            print(f"Current Model: {data['model_type']}")
            print(f"Model Loaded: {data['model_loaded']}")
            print(f"Scaler Loaded: {data['scaler_loaded']}")
            print(f"Features Available: {data['feature_names_available']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Flask app. Is it running on http://localhost:5000?")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def get_model_info():
    """Get available models info"""
    try:
        response = requests.get(f'{BASE_URL}/api/model-info', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_header("AVAILABLE MODELS")
            print(f"Current Model: {data['current_model']}")
            print(f"\nPyTorch Model:")
            print(f"  Available: {data['available_models']['pytorch']['available']}")
            print(f"  Path: {data['available_models']['pytorch']['path']}")
            print(f"\nCatBoost Model:")
            print(f"  Available: {data['available_models']['catboost']['available']}")
            print(f"  Path: {data['available_models']['catboost']['path']}")
            return data['available_models']
        else:
            print(f"Failed to get model info: {response.status_code}")
            return None
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None

def switch_model(model_type):
    """Switch to specified model"""
    if model_type not in ['pytorch', 'catboost']:
        print(f"Invalid model type: {model_type}. Use 'pytorch' or 'catboost'")
        return False
    
    try:
        print(f"\nSwitching to {model_type} model...")
        response = requests.post(
            f'{BASE_URL}/api/switch-model',
            json={'model_type': model_type},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_header(f"SWITCHED TO {model_type.upper()}")
            print(f"Message: {data.get('message', 'N/A')}")
            print(f"Current Model: {data['current_model']}")
            print(f"Model Loaded: {data['model_loaded']}")
            print(f"Scaler Loaded: {data['scaler_loaded']}")
            return True
        else:
            print(f"Switch failed: {response.status_code}")
            print(response.json())
            return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_prediction(image_path):
    """Test prediction with an image"""
    try:
        if not Path(image_path).exists():
            print(f"ERROR: Image file not found: {image_path}")
            return False
        
        print(f"\nTesting prediction with: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f'{BASE_URL}/api/predict', files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print_header("PREDICTION RESULT")
            print(f"GSM Prediction: {data['gsm_prediction']:.2f} g/m²")
            print(f"Confidence: {data['confidence']}")
            print(f"Timestamp: {data['timestamp']}")
            print(f"Filename: {data['filename']}")
            return True
        else:
            print(f"Prediction failed: {response.status_code}")
            print(response.json())
            return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def main():
    """Main function"""
    print_header("GSM FABRIC PREDICTION - MODEL SWITCHING UTILITY")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python model_switch_utility.py health          - Check app health")
        print("  python model_switch_utility.py info            - Get available models")
        print("  python model_switch_utility.py pytorch          - Switch to PyTorch model")
        print("  python model_switch_utility.py catboost         - Switch to CatBoost model")
        print("  python model_switch_utility.py test <image>    - Test prediction with image")
        print("\nExample:")
        print("  python model_switch_utility.py health")
        print("  python model_switch_utility.py switch pytorch")
        print("  python model_switch_utility.py test sample.jpg")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'health':
        check_health()
    
    elif command == 'info':
        get_model_info()
    
    elif command in ['pytorch', 'catboost']:
        switch_model(command)
        check_health()
    
    elif command == 'test':
        if len(sys.argv) < 3:
            print("Usage: python model_switch_utility.py test <image_path>")
            return
        image_path = sys.argv[2]
        test_prediction(image_path)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'health', 'info', 'pytorch', 'catboost', or 'test <image>'")

if __name__ == '__main__':
    main()
