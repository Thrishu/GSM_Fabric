# GSM Fabric Flask Application

## Installation & Setup

### Option 1: Using run_flask.bat (Recommended for Windows)
```bash
cd flask_app
run_flask.bat
```

This will automatically:
1. Install all required dependencies (including timm, torch, torchvision)
2. Start the Flask application

### Option 2: Manual Setup
```bash
cd flask_app
pip install -r requirements.txt
python app.py
```

### Option 3: Using Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

## Important Dependencies

The Flask app requires:
- **timm**: For loading the EfficientNet-B3 model (MUST HAVE)
- **torch**: PyTorch for model inference
- **torchvision**: For image preprocessing utilities
- **opencv-python**: For image processing
- **flask**: Web framework

## Model Architecture

The Flask app uses:
- **Backbone**: EfficientNet-B3 (via timm library)
- **Training Data**: ~1000 fabric images with GSM values (85-297 g/m²)
- **Prediction Range**: 85-297 g/m² (reasonable range)
- **Image Input Size**: 352x352 pixels

## API Usage

### Endpoint: POST /api/predict

```bash
curl -X POST -F "image=@test_image.jpg" http://127.0.0.1:5000/api/predict
```

**Response:**
```json
{
  "success": true,
  "gsm_prediction": 133.50,
  "error": null
}
```

## Troubleshooting

### Error: "No module named 'timm'"
- Make sure you've installed requirements: `pip install -r requirements.txt`
- Or run: `pip install timm`

### Error: "Model not loaded"
- Check that `Model/best_model (1).pt` exists
- Check that the model file is not corrupted

### Predictions are still low (< 0.1)?
- This was caused by using `torchvision.models.efficientnet_b3` instead of `timm.create_model()`
- The fix has been applied to `app.py` - use the updated version

## Expected Behavior

After fix:
- Predictions should return values in range **85-297 g/m²**
- Example: `133.50 g/m²` (not `0.06 g/m²`)
- Predictions may vary slightly between similar images

## Testing

Run the test script:
```bash
python test_flask_model.py
```

This will:
1. Load the model
2. Find test images
3. Make predictions
4. Validate the prediction range
