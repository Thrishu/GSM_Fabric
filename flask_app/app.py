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

# Flask app initialization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Model and scaler paths
MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'best_model (1).pt'  # Using gsm_regressor.pt
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'

# Global variables for model and scaler
model = None
scaler = None
feature_names = None

def load_model_and_scaler():
    """Load the PyTorch CNN model and scaler from checkpoint."""
    global model, scaler, feature_names
    
    try:
        import torch
        import torch.nn as nn
        import timm
        
        # Load model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        print("Loading PyTorch checkpoint...")
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
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
        print(f"[OK] Model loaded successfully from: {MODEL_PATH}")
        print(f"[OK] Scaler loaded successfully")
        return True
    
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
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
        
        # Extract features using the existing function
        features_dict = extract_all_fabric_features_single_image(img_bgr)
        
        if features_dict is None:
            return None, "Failed to extract features"
        
        return features_dict, None
    
    except Exception as e:
        return None, f"Feature extraction error: {str(e)}"

def predict_gsm(features_dict, image_path):
    """Predict GSM value from image using PyTorch CNN model."""
    try:
        import torch
        
        if model is None:
            return None, "Model not loaded"
        
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
        
        print(f"[DEBUG] Raw prediction: {prediction:.2f} g/m²")
        return float(prediction), None
    
    except Exception as e:
        import traceback
        print(f"Prediction error details: {traceback.format_exc()}")
        return None, f"Prediction error: {str(e)}"

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
