
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.preprocessing import RobustScaler
from PIL import Image
from torchvision import models, transforms
import cv2

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = '../Model/gsm_regressor.pt'  # Path relative to flask_app/
DATA_PATH = '../augmented_features_dataset/dataset_train.csv' # Path to training data for scaler fitting

# Add parent directory to path to import extraction script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from extract_fabric_features import (
        extract_thread_count, extract_yarn_density, extract_thread_spacing,
        extract_texture_features, extract_frequency_features, 
        extract_structure_features, extract_edge_features, extract_color_features
    )
except ImportError:
    print("Warning: extract_fabric_features.py not found in parent directory. Copy it to the same folder if running standalone.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Model Definition (Must match training architecture) ---
class HybridGSMPredictor(nn.Module):
    def __init__(self, num_features, dropout=0.5):
        super(HybridGSMPredictor, self).__init__()
        # Pre-trained EfficientNet-B3 backbone
        efficientnet = models.efficientnet_b3(weights=None) # Weights loaded from .pth
        
        # Remove classifier head
        self.cnn_features = nn.Sequential(*list(efficientnet.children())[:-1])
        cnn_feature_size = 1536

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
        cnn_out = self.cnn_features(images)
        cnn_out = torch.flatten(cnn_out, 1)
        feat_out = self.feature_branch(features)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        output = self.fusion(combined)
        return output.squeeze()

# --- Global State ---
model = None
scaler = None
feature_columns = []

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_resources():
    global model, scaler, feature_columns
    
    # 1. Load Data for Scaler
    print("Loading dataset for scaler fitting...")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Identify feature columns (exclude metadata)
        meta_cols = ['image_name', 'gsm', 'source', 'augmentation', 'original_image', 'split']
        feature_columns = [col for col in df.columns if col not in meta_cols]
        
        # Handle missing/zero-var same as training
        for col in feature_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        scaler = RobustScaler()
        scaler.fit(df[feature_columns])
        print(f"Scaler fitted on {len(feature_columns)} features.")
    else:
        print(f"Error: Data file not found at {DATA_PATH}. Feature scaling will fail.")
        return

    # 2. Load Model
    print("Loading PyTorch model...")
    if os.path.exists(MODEL_PATH):
        try:
            model = HybridGSMPredictor(num_features=len(feature_columns))
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Check for model_state key (custom checkpoint format)
                if 'model_state' in checkpoint:
                    model.load_state_dict(checkpoint['model_state'])
                # Check for state_dict key
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                # Otherwise try loading the dict directly as state_dict
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Checkpoint is the model object itself
                model = checkpoint
            
            model.to(device)
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
             print(f"Failed to load model: {e}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")

def extract_features_from_image(img_path):
    # This mimics the logic in extract_fabric_features.process_single_image
    # But runs sequentially for a single image
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = {}
    
    try:
        features.update(extract_thread_count(img_gray, direction='horizontal'))
        features.update(extract_thread_count(img_gray, direction='vertical'))
    except: pass
    
    try: features.update(extract_yarn_density(img_gray))
    except: pass
    
    try: features.update(extract_thread_spacing(img_gray))
    except: pass
    
    try: features.update(extract_texture_features(img_gray))
    except: pass
    
    try: features.update(extract_frequency_features(img_gray))
    except: pass
    
    try: features.update(extract_structure_features(img_gray))
    except: pass
    
    try: features.update(extract_edge_features(img_gray))
    except: pass
    
    try: features.update(extract_color_features(img))
    except: pass
    
    return features

def preprocess_for_model(img_path, features_dict):
    # 1. Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # Batch dimension
    
    # 2. Feature Preprocessing
    # Ensure all columns exist, fill missing with 0 (or some default if scaler expects it)
    # The scaler expects specific columns in order
    feature_vector = []
    
    # We must construct the vector in agreement with 'feature_columns'
    for col in feature_columns:
        val = features_dict.get(col, 0.0) # Default to 0 if extraction failed/missing
        if pd.isna(val): val = 0.0
        feature_vector.append(val)
        
    feature_array = np.array([feature_vector])
    scaled_features = scaler.transform(feature_array)
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
    
    return image_tensor, features_tensor

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in request.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected.")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Run Prediction
            try:
                print(f"Starting processing for {filename}...")
                
                # 1. Extract
                print("Extracting features...")
                raw_features = extract_features_from_image(filepath)
                if not raw_features:
                    print("Feature extraction failed.")
                    return render_template('index.html', error="Failed to process image.")
                print(f"Extracted {len(raw_features)} features.")
                
                # 2. Preprocess
                print("Preprocessing image and features...")
                
                # DIAGNOSTIC: Check feature matching
                missing_feats = []
                for col in feature_columns:
                    if col not in raw_features:
                        missing_feats.append(col)
                print(f"Model expects {len(feature_columns)} features. Missing in extraction: {len(missing_feats)}")
                if len(missing_feats) > 0:
                    print(f"First 5 missing: {missing_feats[:5]}")
                
                img_t, feat_t = preprocess_for_model(filepath, raw_features)
                
                # 3. Predict output
                print("Running model inference...")
                model.eval()  # FORCE EVAL MODE
                print(f"Model training mode: {model.training}")
                with torch.no_grad():
                    prediction = model(img_t, feat_t).item()
                print(f"Prediction: {prediction}")
                
                return render_template('index.html', 
                                       prediction=f"{prediction:.2f} GSM", 
                                       image_url=url_for('static', filename=f'uploads/{filename}'),
                                       features=raw_features)
            except Exception as e:
                print(f"Error during processing: {e}")
                return render_template('index.html', error=str(e))
                
    return render_template('index.html')

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    # Handle base64 image from camera
    import base64
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
        
    image_data = data['image'].split(',')[1]
    filename = 'camera_capture.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    with open(filepath, "wb") as fh:
        fh.write(base64.b64decode(image_data))
    
    try:
        raw_features = extract_features_from_image(filepath)
        img_t, feat_t = preprocess_for_model(filepath, raw_features)
        with torch.no_grad():
            prediction = model(img_t, feat_t).item()
        
        return jsonify({'gsm': f"{prediction:.2f}", 'features': raw_features})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize
load_resources()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
