# Quick Start Guide - GSM Fabric Predictor

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
cd flask_app
pip install -r requirements.txt
```

### Step 2: Run Setup Script

```bash
python setup.py
```

This script will:
- ✅ Check Python version
- ✅ Verify all dependencies
- ✅ Check model files
- ✅ Create necessary directories
- ✅ Create scaler from training data (if needed)

### Step 3: Start the Application

```bash
python app.py
```

You should see:
```
 * Running on http://localhost:5000
 * Press CTRL+C to quit
```

### Step 4: Open in Browser

Visit: **http://localhost:5000**

---

## 📸 How to Use

### Upload an Image
1. Click the **Upload Image** tab
2. Click or drag-drop a fabric microscopy image
3. Wait for prediction
4. See GSM value and confidence

### Take a Photo
1. Click the **Camera Capture** tab
2. Click **Start Camera**
3. Position fabric in view
4. Click **Capture Photo**
5. Get instant prediction

### Process Multiple Images
1. Click the **Batch Upload** tab
2. Select multiple image files
3. Wait for processing
4. View all predictions at once

---

## 🔧 Common Issues

### "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```bash
pip install -r requirements.txt
```

### "Model file not found"

**Solution:**
1. Ensure `gsm_regressor.pt` is in `../Model/`
2. Copy the file:
```bash
cp ../Model/gsm_regressor.pt flask_app/
```

### "Scaler not loaded"

**Solution:**
The scaler will auto-create from training data. Just run:
```bash
python setup.py
```

### Camera not working

**Solution:**
- Give browser permission to access camera
- Try a different browser (Chrome recommended)
- Check camera is not in use by another app

---

## 📊 Expected Outputs

When you upload/capture a fabric image:

```
GSM Value: 150.25 g/m²
Confidence: High
Timestamp: 2024-01-17 10:30:00
```

**GSM Range (Typical Fabrics):**
- Lightweight: 50-100 g/m²
- Standard: 100-200 g/m²
- Heavy: 200-300+ g/m²

---

## 🎯 API Usage Examples

### Using cURL

Single prediction:
```bash
curl -X POST -F "image=@fabric.jpg" http://localhost:5000/api/predict
```

Response:
```json
{
  "success": true,
  "gsm_prediction": 150.25,
  "confidence": "High"
}
```

### Using Python Requests

```python
import requests

# Single image
with open('fabric.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/predict',
        files={'image': f}
    )
    print(response.json())

# Multiple images
with open('fabric1.jpg', 'rb') as f1, open('fabric2.jpg', 'rb') as f2:
    response = requests.post(
        'http://localhost:5000/api/batch-predict',
        files=[('images', f1), ('images', f2)]
    )
    print(response.json())
```

### Using JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('image', imageFile);

fetch('/api/predict', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 📁 File Structure

```
flask_app/
├── app.py                      # Main application
├── setup.py                    # Setup script
├── requirements.txt            # Dependencies
├── extract_fabric_features.py  # Feature extraction
├── templates/
│   └── index.html             # Web UI
├── uploads/                    # Uploaded images
└── README.md                   # Full documentation
```

---

## 🔍 Troubleshooting Commands

Check app is running:
```bash
curl http://localhost:5000/api/health
```

Should return:
```json
{"status": "healthy", "model_loaded": true}
```

View Flask logs:
```bash
python -u app.py  # Unbuffered output
```

Kill app on port 5000 (if stuck):
```bash
lsof -i :5000  # Find process
kill -9 <PID>  # Kill it
```

---

## 💡 Tips

1. **Best Results:**
   - Use clear, well-lit microscopy images
   - Ensure fabric fills most of the frame
   - Use proper magnification

2. **Batch Processing:**
   - Upload 10+ images for efficiency
   - Results show success rate and errors

3. **API Integration:**
   - Check health endpoint before sending requests
   - Handle error responses (400, 500)
   - Max file size is 16MB

4. **Performance:**
   - First prediction slower (model loading)
   - Subsequent predictions are faster
   - ~2-5 seconds per image

---

## 📞 Next Steps

After confirming everything works:

1. **Customize configuration** in `app.py`
2. **Add authentication** for production
3. **Deploy to cloud** (Heroku, AWS, etc.)
4. **Set up monitoring** and logging
5. **Create API documentation** (Swagger)

---

## 🎓 Learning Resources

- **Flask**: https://flask.palletsprojects.com/
- **CatBoost**: https://catboost.ai/
- **OpenCV**: https://opencv.org/
- **Scikit-learn**: https://scikit-learn.org/

---

**Ready to predict fabric GSM? Start the app and visit http://localhost:5000** 🚀
