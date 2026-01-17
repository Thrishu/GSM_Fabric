# 🧵 GSM Fabric Predictor - Flask Application

A web application that predicts fabric weight (GSM - Grams per Square Meter) from microscopy images using deep learning and computer vision feature extraction.

## Features

- 📤 **Upload Images**: Upload fabric microscopy images directly
- 📷 **Camera Capture**: Take photos directly from your device's camera
- 📁 **Batch Processing**: Process multiple images at once
- 🎯 **Real-time Predictions**: Get GSM predictions instantly
- 📊 **Confidence Scores**: Receive confidence levels with predictions

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- A trained CatBoost regressor model (`gsm_regressor.pt`)
- Feature scaler (`scaler.pkl`)

## Installation

### 1. Navigate to the Flask app directory

```bash
cd flask_app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare model files

Ensure these files exist in the `../Model/` directory:
- `gsm_regressor.pt` - Trained CatBoost regressor model
- `scaler.pkl` - Feature scaling object (created automatically if not present)

If `scaler.pkl` doesn't exist, it will be automatically created from the training dataset.

### 4. Update model path (if needed)

Edit `app.py` and update these lines if your model is in a different location:

```python
MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'
```

## Usage

### Starting the application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Web Interface

1. **Upload Tab**: 
   - Click or drag-drop image files
   - Supported formats: PNG, JPG, JPEG, BMP, TIFF
   - Max file size: 16MB

2. **Camera Tab**:
   - Click "Start Camera"
   - Capture a photo
   - Get instant prediction

3. **Batch Tab**:
   - Select multiple image files
   - Process all at once
   - Download results

## API Endpoints

### Single Image Prediction

**POST** `/api/predict`

Request:
```bash
curl -X POST -F "image=@image.jpg" http://localhost:5000/api/predict
```

Response:
```json
{
  "success": true,
  "gsm_prediction": 150.25,
  "confidence": "High",
  "timestamp": "2024-01-17T10:30:00",
  "filename": "image_20240117_103000.jpg"
}
```

### Batch Prediction

**POST** `/api/batch-predict`

Request:
```bash
curl -X POST -F "images=@image1.jpg" -F "images=@image2.jpg" http://localhost:5000/api/batch-predict
```

Response:
```json
{
  "total_processed": 2,
  "successful": 2,
  "results": [
    {
      "filename": "image1.jpg",
      "gsm_prediction": 150.25,
      "success": true
    },
    {
      "filename": "image2.jpg",
      "gsm_prediction": 175.50,
      "success": true
    }
  ]
}
```

### Health Check

**GET** `/api/health`

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

## Model Information

### Input
- **Images**: 512x512 preprocessed fabric microscopy images
- **Features**: 64 extracted computer vision features:
  - Thread count (weft/warp)
  - Yarn density and spacing
  - Texture features (energy, entropy, contrast)
  - Frequency domain features (FFT)
  - Structural features (edges, gradients)
  - Color features

### Output
- **GSM Value**: Predicted fabric weight in grams per square meter
- **Confidence Level**: High/Medium based on prediction range

### Model Type
- **Algorithm**: CatBoost Regressor (Gradient Boosting)
- **Training Data**: ~1000 augmented fabric microscopy images
- **Target Accuracy**: ±5 GSM prediction error

## Feature Extraction Pipeline

The application extracts the following feature categories:

1. **Thread Features**
   - Weft/Warp count
   - Thread spacing (average & std)

2. **Yarn Features**
   - Average yarn area
   - Yarn standard deviation
   - Yarn density

3. **Texture Features**
   - Energy, Entropy
   - Contrast, Homogeneity

4. **Frequency Features**
   - FFT magnitude statistics
   - Power spectrum

5. **Structural Features**
   - Edge density
   - Laplacian variance
   - Sobel magnitude

6. **Edge Features**
   - Line count (Hough)
   - Edge connectivity

7. **Color Features**
   - RGB channel statistics

## File Structure

```
flask_app/
├── app.py                           # Main Flask application
├── extract_fabric_features.py       # Feature extraction module
├── requirements.txt                 # Python dependencies
├── templates/
│   └── index.html                   # Web interface
├── uploads/                         # Uploaded images (auto-created)
└── README.md                        # This file
```

## Troubleshooting

### Model not loading

**Error**: `FileNotFoundError: Model file not found`

**Solution**: 
- Ensure `gsm_regressor.pt` exists in `../Model/`
- Check the MODEL_PATH variable in app.py

### Scaler not found

**Error**: `Scaler not found`

**Solution**:
- The app will create one automatically from the training dataset
- Ensure `data/augmented_features_dataset/dataset_train.csv` exists

### Camera not working

**Issue**: Camera access denied

**Solution**:
- Grant camera permissions in your browser
- Use HTTPS in production (Flask dev server uses HTTP)
- Try a different browser

### Image processing fails

**Error**: `Failed to extract features`

**Solution**:
- Ensure image is a valid microscopy image
- Check image quality (clear, well-lit)
- Try resizing the image

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t gsm-predictor .
docker run -p 5000:8000 gsm-predictor
```

## Configuration

Edit `app.py` to customize:

```python
# Max upload file size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Model paths
MODEL_PATH = Path(__file__).parent.parent / 'Model' / 'gsm_regressor.pt'
SCALER_PATH = Path(__file__).parent.parent / 'Model' / 'scaler.pkl'

# Server port (in main)
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Performance Notes

- First prediction takes longer (model loading)
- Subsequent predictions are faster
- Batch processing is faster per image than individual uploads
- Camera capture works best in well-lit environments

## Development

### Enable debug mode

```python
app.run(debug=True)
```

### Check logs

```bash
# View all requests
python -u app.py  # Unbuffered output
```

## License

Proprietary - Fabric GSM Prediction System

## Support

For issues or questions, contact the development team.

## Future Enhancements

- [ ] Multiple camera input support
- [ ] Real-time streaming predictions
- [ ] Advanced analytics dashboard
- [ ] Model retraining interface
- [ ] Export predictions to CSV/Excel
- [ ] Integration with lab management systems
- [ ] Mobile app version
