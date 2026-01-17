"""
Setup script for GSM Fabric Predictor Flask App
Prepares the environment and validates model files
"""

import os
import sys
import pickle
import json
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Requires 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    required_packages = [
        'flask', 'numpy', 'pandas', 'cv2', 'sklearn', 'scipy', 'catboost', 'torch', 'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    return len(missing) == 0

def check_model_files():
    """Check if model files exist."""
    print("\nChecking model files...")
    
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    model_dir = parent_dir / 'Model'
    
    model_path = model_dir / 'gsm_regressor.pt'
    scaler_path = model_dir / 'scaler.pkl'
    
    model_exists = model_path.exists()
    scaler_exists = scaler_path.exists()
    
    print(f"Model directory: {model_dir}")
    
    if model_exists:
        size = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file exists ({size:.2f} MB)")
    else:
        print(f"❌ Model file NOT found: {model_path}")
    
    if scaler_exists:
        size = scaler_path.stat().st_size / 1024
        print(f"✅ Scaler file exists ({size:.2f} KB)")
    else:
        print(f"⚠️  Scaler file NOT found (will be created automatically)")
    
    return model_exists

def check_dataset_files():
    """Check if training dataset exists for creating scaler."""
    print("\nChecking dataset files...")
    
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    
    dataset_path = parent_dir / 'data' / 'augmented_features_dataset' / 'dataset_train.csv'
    
    if dataset_path.exists():
        print(f"✅ Training dataset found: {dataset_path}")
        return True
    else:
        print(f"⚠️  Training dataset NOT found: {dataset_path}")
        print("   (Will be used to create scaler if needed)")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    script_dir = Path(__file__).parent
    
    dirs_to_create = [
        script_dir / 'uploads',
        script_dir / 'templates',
        script_dir / 'static'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(exist_ok=True)
        print(f"✅ {dir_path.name}/")

def create_scaler():
    """Create scaler from training dataset if it doesn't exist."""
    print("\nTrying to create scaler from training data...")
    
    try:
        from sklearn.preprocessing import RobustScaler
        import pandas as pd
        
        script_dir = Path(__file__).parent
        parent_dir = script_dir.parent
        
        dataset_path = parent_dir / 'data' / 'augmented_features_dataset' / 'dataset_train.csv'
        scaler_path = parent_dir / 'Model' / 'scaler.pkl'
        
        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            return False
        
        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        # Get feature columns
        meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
        feature_cols = [col for col in df.columns if col not in meta_cols]
        
        print(f"Found {len(feature_cols)} features")
        
        # Create and fit scaler
        scaler = RobustScaler()
        scaler.fit(df[feature_cols])
        
        # Save scaler
        os.makedirs(parent_dir / 'Model', exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✅ Scaler created and saved: {scaler_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error creating scaler: {str(e)}")
        return False

def generate_config():
    """Generate configuration file."""
    print("\nGenerating configuration...")
    
    script_dir = Path(__file__).parent
    config = {
        'app': {
            'debug': False,
            'host': '0.0.0.0',
            'port': 5000,
            'max_file_size_mb': 16
        },
        'model': {
            'model_path': str(Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'),
            'scaler_path': str(Path(__file__).parent.parent / 'Model' / 'scaler.pkl')
        },
        'features': {
            'target_size': [512, 512],
            'allowed_extensions': ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        }
    }
    
    config_path = script_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved: {config_path}")

def print_summary(checks):
    """Print summary of checks."""
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    all_good = all(checks.values())
    
    for check, passed in checks.items():
        status = "✅" if passed else "⚠️ "
        print(f"{status} {check}")
    
    print("="*60)
    
    if all_good:
        print("\n🎉 All checks passed! Ready to run:")
        print("   python app.py")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("\nQuick fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy model file to ../Model/gsm_regressor.pt")
        print("3. Run setup again: python setup.py")

def main():
    """Run all setup checks."""
    print("🧵 GSM Fabric Predictor - Setup Script")
    print("="*60)
    
    checks = {}
    
    # Run checks
    checks['Python Version'] = check_python_version()
    checks['Dependencies'] = check_dependencies()
    checks['Model Files'] = check_model_files()
    checks['Dataset Files'] = check_dataset_files()
    
    # Create directories
    create_directories()
    
    # Try to create scaler
    if not (Path(__file__).parent.parent / 'Model' / 'scaler.pkl').exists():
        create_scaler()
    
    # Generate config
    generate_config()
    
    # Print summary
    print_summary(checks)

if __name__ == '__main__':
    main()
