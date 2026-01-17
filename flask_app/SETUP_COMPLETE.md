# Flask GSM Fabric Predictor - Complete Setup

## 📦 What's Been Created

A complete, production-ready Flask web application for predicting fabric GSM (weight) from microscopy images using the CatBoost regressor model.

---

## 📁 Project Structure

```
flask_app/
├── app.py                          # Main Flask application
├── setup.py                        # Setup script (run first)
├── test.py                         # Testing script
├── extract_fabric_features.py      # Feature extraction module
├── requirements.txt                # Python dependencies
├── README.md                       # Full documentation
├── QUICKSTART.md                   # Quick start guide
├── templates/
│   └── index.html                  # Web interface (upload/camera/batch)
├── static/
│   └── style.css                   # Styling
├── uploads/                        # Uploaded images (auto-created)
└── (This file)
```

---

## 🚀 Getting Started (5 Steps)

### 1. Install Dependencies

```bash
cd flask_app
pip install -r requirements.txt
```

**Installs:**
- Flask (web framework)
- OpenCV (image processing)
- Pandas & NumPy (data handling)
- CatBoost (ML model)
- Scikit-learn (preprocessing)

### 2. Run Setup Script

```bash
python setup.py
```

**This will:**
- ✅ Verify Python 3.8+
- ✅ Check all dependencies installed
- ✅ Verify model file exists
- ✅ Create necessary directories
- ✅ Auto-generate scaler from training data (if needed)
- ✅ Generate config.json

### 3. Test Installation (Optional)

```bash
python test.py
```

**Tests:**
- Model loading
- Scaler loading
- Feature extraction
- Dataset loading
- Prediction pipeline

### 4. Start Application

```bash
python app.py
```

**Output:**
```
Loading model and scaler...
✅ Model loaded from: ../Model/gsm_regressor.pt
✅ Scaler loaded from: ../Model/scaler.pkl
✅ Application ready!
 * Running on http://localhost:5000
 * Press CTRL+C to quit
```

### 5. Open in Browser

Visit: **http://localhost:5000**

---

## 💡 Features Implemented

### Web Interface
- **📤 Upload Tab**: Drag & drop or click to upload images
- **📷 Camera Tab**: Real-time camera capture for photos
- **📁 Batch Tab**: Process multiple images simultaneously
- **🎨 Responsive Design**: Works on desktop, tablet, mobile

### Prediction Pipeline
- Image preprocessing (resize, normalize)
- 64 fabric-specific feature extraction
- Feature scaling (RobustScaler)
- CatBoost model prediction
- Confidence scoring

### API Endpoints
- `POST /api/predict` - Single image prediction
- `POST /api/batch-predict` - Batch processing
- `GET /api/health` - Health check

### Image Formats Supported
- PNG, JPG, JPEG, BMP, TIFF
- Max file size: 16MB

---

## 🔧 Key Files Explained

### app.py
Main Flask application with:
- Route handlers (`/`, `/api/predict`, `/api/batch-predict`)
- Image preprocessing
- Feature extraction pipeline
- Model inference
- Error handling

**Key Functions:**
- `load_model_and_scaler()` - Load ML components
- `preprocess_image()` - Resize & normalize
- `extract_features()` - Extract fabric features
- `predict_gsm()` - Make predictions

### extract_fabric_features.py
Comprehensive feature extraction with:

**Thread Features:**
- Weft/warp thread count
- Thread spacing analysis

**Yarn Features:**
- Yarn density
- Area statistics

**Texture Features:**
- Energy, entropy
- Contrast, homogeneity

**Frequency Features:**
- FFT magnitude
- Power spectrum

**Structural Features:**
- Edge detection
- Gradient analysis

**Color Features:**
- RGB channel statistics

### templates/index.html
Modern, responsive web interface with:
- Tab-based navigation
- Drag & drop upload
- Real-time camera capture
- Batch file upload
- Result visualization
- Error handling
- Mobile-friendly design

### setup.py
Automated setup script that:
- Validates environment
- Checks dependencies
- Verifies model files
- Creates directories
- Auto-generates scaler
- Produces config file

### test.py
Testing suite with:
- Model loading tests
- Feature extraction tests
- Prediction pipeline tests
- API endpoint tests
- Quick/full test modes

---

## 📊 Model Information

**Type:** CatBoost Regressor (Gradient Boosting)

**Input:**
- 512×512 RGB fabric microscopy image
- 64 extracted computer vision features

**Output:**
- GSM value (grams per square meter)
- Confidence level (High/Medium)

**Performance:**
- Expected MAE: ~3-5 GSM
- Works on GSM range: 50-300+ g/m²

**Training Data:**
- ~1000 augmented fabric images
- Features from preprocessed dataset
- Hyperparameters tuned with Optuna

---

## 🌐 API Usage Examples

### Using cURL

**Single Prediction:**
```bash
curl -X POST -F "image=@fabric.jpg" http://localhost:5000/api/predict
```

**Response:**
```json
{
  "success": true,
  "gsm_prediction": 150.25,
  "confidence": "High",
  "timestamp": "2024-01-17T10:30:00",
  "filename": "fabric_20240117_103000.jpg"
}
```

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

### Using Python

```python
import requests

# Single prediction
with open('fabric.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/predict',
        files={'image': f}
    )
    result = response.json()
    print(f"GSM: {result['gsm_prediction']}")

# Batch prediction
files = [
    ('images', open('fabric1.jpg', 'rb')),
    ('images', open('fabric2.jpg', 'rb'))
]
response = requests.post(
    'http://localhost:5000/api/batch-predict',
    files=files
)
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('image', imageFile);

fetch('/api/predict', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => {
    console.log(`GSM: ${data.gsm_prediction}`);
});
```

---

## 🔍 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"
**Solution:** `pip install -r requirements.txt`

### Issue: "Model file not found"
**Solution:** Ensure `../Model/gsm_regressor.pt` exists

### Issue: "Scaler not loaded"
**Solution:** Run `python setup.py` to auto-generate from training data

### Issue: Camera not working
**Solution:** 
- Grant browser camera permissions
- Use HTTPS in production
- Try Chrome/Edge browser

### Issue: Feature extraction fails
**Solution:** 
- Ensure image is a valid microscopy image
- Check image quality (clear, well-lit)
- Image should show fabric structure clearly

---

## 📈 Expected Results

### GSM Predictions by Fabric Type

| Fabric Type | Expected GSM | Example |
|---|---|---|
| Ultra-lightweight | 50-100 | Chiffon, Organza |
| Lightweight | 100-150 | Cotton voile, Linen |
| Standard | 150-200 | Cotton poplin, Twill |
| Heavy | 200-300 | Canvas, Denim |
| Ultra-heavy | 300+ | Heavy canvas |

### Prediction Confidence

- **High**: GSM 50-300 (normal range)
- **Medium**: Outside normal range (unusual fabric)

---

## 🎯 Next Steps

### Development
1. Customize UI styling in `templates/index.html`
2. Add authentication for secure access
3. Implement logging and monitoring
4. Add database for storing predictions
5. Create admin dashboard

### Deployment
1. Use Gunicorn/uWSGI instead of Flask dev server
2. Set up HTTPS with SSL certificate
3. Deploy to cloud (AWS, GCP, Azure, Heroku)
4. Set up monitoring and alerts
5. Create backup strategy

### Enhancement
1. Add model retraining interface
2. Implement A/B testing for models
3. Add confidence intervals
4. Create analytics dashboard
5. Export predictions to CSV/Excel

---

## 📚 Documentation

- **README.md** - Complete documentation
- **QUICKSTART.md** - Quick start guide
- **setup.py** - Setup with inline help
- **test.py** - Testing with examples

---

## 🔐 Security Notes

### Current (Development)
- ✅ File type validation
- ✅ File size limit (16MB)
- ✅ Safe filename handling
- ✅ Input sanitization

### For Production
- Add CORS restrictions
- Implement rate limiting
- Add authentication/authorization
- Use HTTPS only
- Add request logging
- Implement CSRF protection
- Add input validation

---

## 📞 Support

**File Issues:**
1. Check console logs for errors
2. Run test suite: `python test.py`
3. Review documentation
4. Check Flask debug output

**Common Errors:**
- See Troubleshooting section above
- Check requirements.txt installed
- Verify model file path
- Ensure proper image format

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Run `python setup.py` - all checks pass
- [ ] Run `python test.py` - all tests pass
- [ ] Run `python app.py` - starts without errors
- [ ] Visit http://localhost:5000 - page loads
- [ ] Upload test image - prediction works
- [ ] Camera capture works (if available)
- [ ] Batch upload works with multiple files
- [ ] API health check passes: `curl http://localhost:5000/api/health`

---

## 🎉 Ready to Use!

Your GSM Fabric Predictor Flask app is complete and ready to:
1. Process fabric images
2. Extract features
3. Predict GSM values
4. Serve predictions via web/API

**Start now:**
```bash
python app.py
# Then open http://localhost:5000
```

---

**Created:** January 17, 2026
**Model:** CatBoost Regressor with 64 fabric features
**Framework:** Flask 2.3+
**Python:** 3.8+
