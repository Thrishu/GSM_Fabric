import os
import sys
import io
import json
import pickle
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64
from datetime import datetime

# Import local feature extraction module
from extract_fabric_features import extract_all_fabric_features_single_image

# Helper functions for log transformation
def inverse_log_transform_gsm(gsm_log):
    """Inverse log transformation: expm1(gsm_log) = exp(gsm_log) - 1"""
    return np.expm1(gsm_log)

# Flask app initialization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Model paths
PYTORCH_MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'best_model (1).pt'
CATBOOST_MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'catboost_baseline.cbm'
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'
FEATURE_SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'feature_scaler.pkl'  # Scaler trained on exact CatBoost features

# Configuration: Select which model to use ('pytorch' or 'catboost')
MODEL_TYPE = 'pytorch'  # Change this to 'catboost' to use CatBoost model

# Global variables for model and scaler
model = None
scaler = None
feature_names = None
current_model_type = None

def load_model_and_scaler():
    """Load model and scaler based on MODEL_TYPE configuration."""
    global model, scaler, feature_names, current_model_type
    
    current_model_type = MODEL_TYPE
    
    if MODEL_TYPE.lower() == 'catboost':
        return load_catboost_model()
    elif MODEL_TYPE.lower() == 'pytorch':
        return load_pytorch_model()
    else:
        print(f"[ERROR] Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'pytorch' or 'catboost'")
        return False

def load_catboost_model():
    """Load CatBoost model and feature scaler."""
    global model, scaler, feature_names
    
    try:
        from catboost import CatBoostRegressor
        
        # Check if model file exists
        if not CATBOOST_MODEL_PATH.exists():
            raise FileNotFoundError(f"CatBoost model file not found: {CATBOOST_MODEL_PATH}")
        
        print(f"Loading CatBoost model from: {CATBOOST_MODEL_PATH}")
        model = CatBoostRegressor()
        model.load_model(str(CATBOOST_MODEL_PATH), format='cbm')
        print(f"[OK] CatBoost model loaded successfully")
        
        # Load feature scaler - this contains the exact features CatBoost was trained on
        if FEATURE_SCALER_PATH.exists():
            with open(FEATURE_SCALER_PATH, 'rb') as f:
                scaler_data = pickle.load(f)
            
            # The pickle file might contain just the scaler or a dict with scaler + feature names
            if isinstance(scaler_data, dict):
                scaler = scaler_data.get('scaler', scaler_data)
                feature_names = scaler_data.get('feature_names', None)
                print(f"[OK] Loaded scaler and feature info from: {FEATURE_SCALER_PATH}")
                if feature_names:
                    print(f"[DEBUG] Scaler trained on {len(feature_names)} features")
                    print(f"[DEBUG] Feature names (first 10): {feature_names[:10]}")
            else:
                scaler = scaler_data
                print(f"[OK] Loaded scaler from: {FEATURE_SCALER_PATH}")
                # Try to get feature count from scaler
                if hasattr(scaler, 'n_features_in_'):
                    print(f"[DEBUG] Scaler expects {scaler.n_features_in_} features")
            
            # If feature names still not loaded, infer from training data using correlation filtering
            if feature_names is None:
                print("[DEBUG] Feature names not in pickle, inferring from training data with correlation filtering...")
                augmented_csv = Path(__file__).parent.parent / 'split_feature_dataset' / 'train' / 'dataset_train.csv'
                if augmented_csv.exists():
                    df = pd.read_csv(augmented_csv)
                    meta_cols = ['image_name', 'gsm', 'source']
                    all_features = [col for col in df.columns if col not in meta_cols]
                    
                    print(f"[DEBUG] Initial features from CSV: {len(all_features)}")
                    
                    # Apply filtering: remove near-constant features (variance < 0.01)
                    variance_threshold = 0.01
                    feature_variance = df[all_features].var()
                    constant_features = feature_variance[feature_variance < variance_threshold].index.tolist()
                    filtered_features = [f for f in all_features if f not in constant_features]
                    
                    print(f"[DEBUG] After removing constant features ({len(constant_features)} removed): {len(filtered_features)}")
                    
                    # Remove highly correlated features (correlation > 0.95) - SAME AS TRAINING
                    correlation_threshold = 0.95
                    corr_matrix = df[filtered_features].corr().abs()
                    upper = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    drop_cols = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
                    feature_names = [f for f in filtered_features if f not in drop_cols]
                    
                    print(f"[DEBUG] After removing correlated features ({len(drop_cols)} removed): {len(feature_names)}")
                    print(f"[DEBUG] Final feature set: {feature_names}")
                else:
                    raise FileNotFoundError(f"Training dataset not found at {augmented_csv}")
        else:
            print(f"[WARNING] Feature scaler not found at: {FEATURE_SCALER_PATH}")
            print(f"[WARNING] Inferring features from training dataset...")
            
            # Load from training dataset and apply same filtering as training
            augmented_csv = Path(__file__).parent.parent / 'split_feature_dataset' / 'train' / 'dataset_train.csv'
            if augmented_csv.exists():
                df = pd.read_csv(augmented_csv)
                meta_cols = ['image_name', 'gsm', 'source']
                all_features = [col for col in df.columns if col not in meta_cols]
                
                print(f"[DEBUG] Initial features from CSV: {len(all_features)}")
                
                # Apply filtering: remove near-constant features
                variance_threshold = 0.01
                feature_variance = df[all_features].var()
                constant_features = feature_variance[feature_variance < variance_threshold].index.tolist()
                filtered_features = [f for f in all_features if f not in constant_features]
                
                print(f"[DEBUG] After removing constant features: {len(filtered_features)}")
                
                # Remove highly correlated features
                correlation_threshold = 0.95
                corr_matrix = df[filtered_features].corr().abs()
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                drop_cols = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
                feature_names = [f for f in filtered_features if f not in drop_cols]
                
                print(f"[DEBUG] After removing correlated features: {len(feature_names)}")
                
                # Create scaler from selected features
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(df[feature_names])
                print(f"[OK] Created scaler from dataset with {len(feature_names)} features")
            else:
                print("[ERROR] Cannot load training dataset")
                return False
        
        print(f"[OK] CatBoost model loaded with {len(feature_names) if feature_names else 'unknown'} features")
        return True
    
    except Exception as e:
        print(f"[ERROR] Error loading CatBoost model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_pytorch_model():
    """Load PyTorch CNN model and scaler from checkpoint."""
    global model, scaler, feature_names
    
    try:
        import torch
        import torch.nn as nn
        import timm
        
        # Load model
        if not PYTORCH_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {PYTORCH_MODEL_PATH}")
        
        print("Loading PyTorch checkpoint...")
        checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location='cpu')
        
        print(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Define model architecture - MUST MATCH CHECKPOINT EXACTLY (uses timm)
        class EfficientNetGSMPredictor(nn.Module):
            """EfficientNet-B3 model for GSM prediction using timm."""
            def __init__(self, backbone_name='efficientnet_b3'):
                super(EfficientNetGSMPredictor, self).__init__()
                # Create backbone with timm (matches Colab notebook)
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=False,
                    num_classes=0,  # Remove classifier
                    global_pool='avg'  # Global average pooling
                )
                # Simple prediction head: 1536 -> 1
                self.head = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.backbone.num_features, 1)
                )
            
            def forward(self, images):
                cnn_out = self.backbone(images)
                output = self.head(cnn_out)
                return output.squeeze()
        
        # Load checkpoint and reconstruct model
        model_loaded = False
        scaler_loaded = False
        
        if isinstance(checkpoint, dict):
            # Check for model_state or model_state_dict
            model_state_dict = None
            feature_cols = None
            
            if 'model_state_dict' in checkpoint:
                print("Found 'model_state_dict' in checkpoint")
                model_state_dict = checkpoint['model_state_dict']
                feature_cols = checkpoint.get('feature_cols', [])
            elif 'model_state' in checkpoint:
                print("Found 'model_state' in checkpoint")
                model_state_dict = checkpoint['model_state']
                # Try to infer feature count from state dict
                # Look for the first Linear layer input size in feature_branch
                for key in model_state_dict.keys():
                    if 'feature_branch.0.weight' in key:
                        # This is the first linear layer: Linear(num_features, 256)
                        feature_cols = list(range(model_state_dict[key].shape[1]))
                        print(f"Inferred {len(feature_cols)} features from state dict")
                        break
            
            if model_state_dict is not None:
                try:
                    print(f"Creating EfficientNet-B3 model (checkpoint has 'backbone' and 'head')...")
                    # Initialize EfficientNet-B3 model matching Colab
                    model = EfficientNetGSMPredictor(backbone_name='efficientnet_b3')
                    
                    # Load state dict with strict=False
                    incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
                    if incompatible_keys.missing_keys:
                        print(f"[WARNING] Missing keys: {len(incompatible_keys.missing_keys)}")
                    if incompatible_keys.unexpected_keys:
                        print(f"[WARNING] Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
                    
                    model.eval()  # Set to evaluation mode
                    print(f"[OK] Model loaded successfully")
                    model_loaded = True
                    
                    # Extract scaler from checkpoint if available
                    if 'scaler' in checkpoint:
                        scaler = checkpoint['scaler']
                        print(f"[OK] Scaler loaded from checkpoint")
                        scaler_loaded = True
                
                except Exception as e:
                    print(f"[ERROR] Error loading model: {e}")
                    import traceback
                    traceback.print_exc()
                    model = None
            else:
                print(f"[WARNING] No model state found in checkpoint. Available keys: {list(checkpoint.keys())}")
                model = None
        else:
            # Try direct model loading
            model = checkpoint
            print(f"[OK] Loaded checkpoint directly")
            model_loaded = True
        
        # Load scaler if not already loaded
        if not scaler_loaded:
            if SCALER_PATH.exists():
                with open(SCALER_PATH, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"[OK] Scaler loaded from: {SCALER_PATH}")
                scaler_loaded = True
            else:
                # Create from dataset
                augmented_csv = Path(__file__).parent.parent / 'data' / 'augmented_features_dataset' / 'dataset_train.csv'
                if augmented_csv.exists():
                    df = pd.read_csv(augmented_csv)
                    meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
                    feature_names = [col for col in df.columns if col not in meta_cols]
                    
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    scaler.fit(df[feature_names])
                    
                    # Save scaler for future use
                    with open(SCALER_PATH, 'wb') as f:
                        pickle.dump(scaler, f)
                    print(f"[OK] Scaler created from dataset and saved")
                    scaler_loaded = True
                else:
                    print("[WARNING] Scaler not found and cannot create from dataset")
        
        # Verify model is properly loaded
        if not model_loaded or model is None:
            print("[ERROR] Model failed to load properly")
            return False
        
        if isinstance(model, dict):
            print("[ERROR] Model is still a dict, cannot be used for predictions")
            print(f"Model type: {type(model)}, keys: {list(model.keys()) if isinstance(model, dict) else 'N/A'}")
            model = None
            return False
        
        print(f"[OK] Model type: {type(model)}")
        print(f"[OK] PyTorch model loaded successfully from: {PYTORCH_MODEL_PATH}")
        print(f"[OK] Scaler loaded successfully")
        return True
    
    except Exception as e:
        print(f"[ERROR] Error loading PyTorch model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(512, 512)):
    """Preprocess image for feature extraction."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize with aspect ratio preservation
        height, width = img.shape[:2]
        scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img_resized = cv2.resize(img, (new_width, new_height))
        
        # Add padding
        top = (target_size[1] - new_height) // 2
        bottom = target_size[1] - new_height - top
        left = (target_size[0] - new_width) // 2
        right = target_size[0] - new_width - left
        
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        return img_padded
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def extract_features(image_path):
    """Extract fabric features from image."""
    try:
        # Preprocess image
        img = preprocess_image(image_path)
        if img is None:
            return None, "Failed to preprocess image"
        
        # Convert back to BGR for cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Extract features using the existing function (with debug=True for troubleshooting)
        features_dict = extract_all_fabric_features_single_image(img_bgr, debug=True)
        
        if features_dict is None:
            return None, "Failed to extract features"
        
        return features_dict, None
    
    except Exception as e:
        return None, f"Feature extraction error: {str(e)}"

def predict_gsm(features_dict, image_path):
    """Predict GSM value using the selected model (PyTorch CNN or CatBoost)."""
    if current_model_type == 'catboost':
        return predict_gsm_catboost(features_dict, image_path)
    elif current_model_type == 'pytorch':
        return predict_gsm_pytorch(features_dict, image_path)
    else:
        return None, f"Unknown model type: {current_model_type}"

def predict_gsm_catboost(features_dict, image_path):
    """Predict GSM value from features using CatBoost model."""
    try:
        if model is None:
            return None, "CatBoost model not loaded"
        
        if scaler is None:
            return None, "Scaler not loaded"
        
        if feature_names is None or len(feature_names) == 0:
            return None, "Feature names not available"
        
        # Create feature vector from extracted features
        # Use mean imputation for missing features
        feature_vector = []
        extracted_features = 0
        missing_features_list = []
        
        for fname in feature_names:
            if fname in features_dict:
                feature_vector.append(features_dict[fname])
                extracted_features += 1
            else:
                # Track missing features
                feature_vector.append(0.0)
                missing_features_list.append(fname)
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        print(f"[DEBUG] Expected {len(feature_names)} features, got {extracted_features} extracted")
        print(f"[DEBUG] Feature vector shape: {feature_vector.shape}")
        print(f"[DEBUG] Feature vector range: [{feature_vector.min():.4f}, {feature_vector.max():.4f}]")
        
        if missing_features_list:
            print(f"[WARNING] Missing {len(missing_features_list)} features: {missing_features_list}")
        
        # Debug: compare with scaler expectations
        if hasattr(scaler, 'n_features_in_'):
            print(f"[DEBUG] Scaler expects {scaler.n_features_in_} features, got {len(feature_names)}")
        
        # Scale features
        try:
            feature_vector_scaled = scaler.transform(feature_vector)
        except Exception as e:
            print(f"[ERROR] Scaler transform error: {str(e)}")
            print(f"[ERROR] Feature names count: {len(feature_names)}")
            print(f"[ERROR] Feature vector shape: {feature_vector.shape}")
            if hasattr(scaler, 'n_features_in_'):
                print(f"[ERROR] Scaler expects {scaler.n_features_in_} features")
            return None, f"Feature scaling error: {str(e)}"
        
        # Make prediction with CatBoost
        prediction = model.predict(feature_vector_scaled)[0]
        
        # Inverse log transform to get actual GSM value
        gsm_prediction = inverse_log_transform_gsm(prediction)
        
        print(f"[DEBUG] CatBoost log prediction: {prediction:.4f}")
        print(f"[DEBUG] CatBoost GSM prediction: {gsm_prediction:.2f} g/m²")
        return float(gsm_prediction), None
    
    except Exception as e:
        import traceback
        print(f"CatBoost prediction error details: {traceback.format_exc()}")
        return None, f"CatBoost prediction error: {str(e)}"

def predict_gsm_pytorch(features_dict, image_path):
    """Predict GSM value from image using PyTorch CNN model."""
    try:
        import torch
        
        if model is None:
            return None, "PyTorch model not loaded"
        
        # Load and preprocess the image
        IMG_SIZE = 352  # Standard EfficientNet input size
        img = cv2.imread(str(image_path))
        if img is None:
            return None, "Failed to load image for prediction"
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize image to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        # Apply ImageNet normalization (standard for pre-trained models)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        print(f"[DEBUG] Image tensor shape: {img_tensor.shape}, range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
        
        # Make prediction (model outputs raw GSM value, no denormalization needed)
        with torch.no_grad():
            prediction = model(img_tensor).item()
        
        print(f"[DEBUG] PyTorch prediction: {prediction:.2f} g/m²")
        return float(prediction), None
    
    except Exception as e:
        import traceback
        print(f"PyTorch prediction error details: {traceback.format_exc()}")
        return None, f"PyTorch prediction error: {str(e)}"

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction."""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIFF'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        file.save(filepath)
        
        # Extract features
        features_dict, feature_error = extract_features(filepath)
        if feature_error:
            os.remove(filepath)
            return jsonify({'error': feature_error}), 400
        
        # Predict GSM
        gsm_prediction, predict_error = predict_gsm(features_dict, filepath)
        if predict_error:
            os.remove(filepath)
            return jsonify({'error': predict_error}), 400
        
        # Return prediction
        result = {
            'success': True,
            'gsm_prediction': gsm_prediction,
            'confidence': 'High' if 50 <= gsm_prediction <= 300 else 'Medium',
            'timestamp': datetime.now().isoformat(),
            'filename': os.path.basename(filepath)
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_type': current_model_type,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names_available': feature_names is not None
    }), 200

@app.route('/api/switch-model', methods=['POST'])
def switch_model():
    """Switch between PyTorch and CatBoost models."""
    global model, scaler, feature_names, current_model_type, MODEL_TYPE
    
    try:
        data = request.get_json()
        new_model_type = data.get('model_type', '').lower()
        
        if new_model_type not in ['pytorch', 'catboost']:
            return jsonify({'error': "Model type must be 'pytorch' or 'catboost'"}), 400
        
        if new_model_type == current_model_type:
            return jsonify({
                'message': f'Already using {new_model_type} model',
                'current_model': current_model_type
            }), 200
        
        # Update global MODEL_TYPE
        MODEL_TYPE = new_model_type
        
        # Reset globals
        model = None
        scaler = None
        feature_names = None
        
        # Load new model
        print(f"\n[SWITCH] Switching from {current_model_type} to {new_model_type}...")
        success = load_model_and_scaler()
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully switched to {new_model_type} model',
                'current_model': current_model_type,
                'model_loaded': model is not None,
                'scaler_loaded': scaler is not None
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to load {new_model_type} model',
                'current_model': current_model_type
            }), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about available and current models."""
    pytorch_available = PYTORCH_MODEL_PATH.exists()
    catboost_available = CATBOOST_MODEL_PATH.exists()
    
    return jsonify({
        'current_model': current_model_type,
        'available_models': {
            'pytorch': {
                'available': pytorch_available,
                'path': str(PYTORCH_MODEL_PATH)
            },
            'catboost': {
                'available': catboost_available,
                'path': str(CATBOOST_MODEL_PATH)
            }
        },
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 200

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from multiple images."""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
                file.save(filepath)
                
                features_dict, _ = extract_features(filepath)
                if features_dict:
                    gsm_pred, _ = predict_gsm(features_dict, filepath)
                    if gsm_pred:
                        results.append({
                            'filename': file.filename,
                            'gsm_prediction': gsm_pred,
                            'success': True
                        })
                    else:
                        results.append({
                            'filename': file.filename,
                            'error': 'Prediction failed',
                            'success': False
                        })
                
                os.remove(filepath)
        
        return jsonify({
            'total_processed': len(files),
            'successful': sum(1 for r in results if r.get('success')),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading model and scaler...")
    if load_model_and_scaler():
        print("[OK] Application ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("[ERROR] Failed to load model. Please check the model files.")
