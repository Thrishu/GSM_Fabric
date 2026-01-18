#!/usr/bin/env python3
"""
Model Comparison Tool - Compare PyTorch and CatBoost predictions on test images
"""

import requests
import json
import time
import sys
from pathlib import Path
from tabulate import tabulate
import statistics

BASE_URL = 'http://localhost:5000'

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")

def switch_model_safe(model_type):
    """Switch model with error handling"""
    try:
        response = requests.post(
            f'{BASE_URL}/api/switch-model',
            json={'model_type': model_type},
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error switching to {model_type}: {str(e)}")
        return False

def predict_image(image_path):
    """Get prediction for an image"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            start = time.time()
            response = requests.post(f'{BASE_URL}/api/predict', files=files, timeout=30)
            elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            return {
                'gsm': data['gsm_prediction'],
                'confidence': data['confidence'],
                'time': elapsed,
                'success': True
            }
        else:
            return {'success': False, 'error': response.json().get('error', 'Unknown error')}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def compare_models(image_paths):
    """Compare PyTorch and CatBoost on multiple images"""
    
    print_header("MODEL COMPARISON TOOL")
    
    if not image_paths:
        print("No images provided. Usage:")
        print("  python model_comparison.py image1.jpg image2.jpg ...")
        return
    
    # Verify images exist
    valid_images = []
    for img_path in image_paths:
        if Path(img_path).exists():
            valid_images.append(img_path)
        else:
            print(f"Warning: Image not found - {img_path}")
    
    if not valid_images:
        print("No valid images found!")
        return
    
    print(f"Testing {len(valid_images)} image(s)\n")
    
    # Results storage
    results = []
    pytorch_times = []
    catboost_times = []
    
    # Test with PyTorch
    print("Testing PyTorch model...")
    if not switch_model_safe('pytorch'):
        print("ERROR: Failed to switch to PyTorch model")
        return
    
    time.sleep(1)
    
    for i, img_path in enumerate(valid_images, 1):
        print(f"  [{i}/{len(valid_images)}] Processing {Path(img_path).name}...", end=' ', flush=True)
        pred = predict_image(img_path)
        print("Done")
        
        if pred['success']:
            pytorch_times.append(pred['time'])
            result = {
                'Image': Path(img_path).name,
                'PyTorch (g/m²)': f"{pred['gsm']:.2f}",
                'PyTorch Time (s)': f"{pred['time']:.3f}",
            }
            results.append(result)
        else:
            print(f"    Error: {pred['error']}")
            results.append({'Image': Path(img_path).name, 'PyTorch': 'ERROR'})
    
    print()
    
    # Test with CatBoost
    print("Testing CatBoost model...")
    if not switch_model_safe('catboost'):
        print("ERROR: Failed to switch to CatBoost model")
        return
    
    time.sleep(1)
    
    for i, img_path in enumerate(valid_images, 1):
        print(f"  [{i}/{len(valid_images)}] Processing {Path(img_path).name}...", end=' ', flush=True)
        pred = predict_image(img_path)
        print("Done")
        
        if pred['success']:
            catboost_times.append(pred['time'])
            for result in results:
                if result['Image'] == Path(img_path).name:
                    result['CatBoost (g/m²)'] = f"{pred['gsm']:.2f}"
                    result['CatBoost Time (s)'] = f"{pred['time']:.3f}"
                    break
        else:
            print(f"    Error: {pred['error']}")
    
    # Print results table
    print_header("PREDICTION RESULTS")
    print(tabulate(results, headers='keys', tablefmt='grid'))
    
    # Print statistics
    print_header("PERFORMANCE STATISTICS")
    
    if pytorch_times:
        stats_data = [
            ['Metric', 'PyTorch', 'CatBoost'],
            ['Avg Time (s)', f"{statistics.mean(pytorch_times):.4f}", f"{statistics.mean(catboost_times):.4f}" if catboost_times else 'N/A'],
            ['Min Time (s)', f"{min(pytorch_times):.4f}", f"{min(catboost_times):.4f}" if catboost_times else 'N/A'],
            ['Max Time (s)', f"{max(pytorch_times):.4f}", f"{max(catboost_times):.4f}" if catboost_times else 'N/A'],
            ['Total Time (s)', f"{sum(pytorch_times):.4f}", f"{sum(catboost_times):.4f}" if catboost_times else 'N/A'],
        ]
        
        if len(pytorch_times) > 1:
            stats_data.append(['Std Dev (s)', f"{statistics.stdev(pytorch_times):.4f}", f"{statistics.stdev(catboost_times):.4f}" if len(catboost_times) > 1 else 'N/A'])
        
        print(tabulate(stats_data[1:], headers=stats_data[0], tablefmt='grid'))
    
    print("\n")
    if catboost_times and pytorch_times:
        speedup = sum(pytorch_times) / sum(catboost_times)
        print(f"CatBoost is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} on average")
    
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python model_comparison.py <image1> [image2] [image3] ...")
        print("\nExample:")
        print("  python model_comparison.py test1.jpg test2.jpg")
        print("\nThis will:")
        print("  1. Switch to PyTorch and test all images")
        print("  2. Switch to CatBoost and test all images")
        print("  3. Compare results and performance")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    compare_models(image_paths)
