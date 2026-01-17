# 🧵 GSM Fabric Predictor - Start Here!

## ⚡ Super Quick Start (5 minutes)

### Step 1: Install Dependencies (1 min)
```bash
cd flask_app
pip install -r requirements.txt
```

### Step 2: Run Setup (1 min)
```bash
python setup.py
```
✅ This validates everything and creates config files

### Step 3: Start App (30 sec)
```bash
python app.py
```
You should see:
```
✅ Model loaded from: ../Model/gsm_regressor.pt
✅ Scaler loaded from: ../Model/scaler.pkl
✅ Application ready!
 * Running on http://localhost:5000
```

### Step 4: Open Browser (30 sec)
🌐 Visit: **http://localhost:5000**

### Step 5: Upload Image (1 min)
- Click "Upload Image" tab
- Drag & drop a fabric microscopy image
- **Boom!** See GSM prediction

---

## 🎯 What You Can Do

```
┌─────────────────────────────────────┐
│  GSM FABRIC PREDICTOR WEB APP       │
├─────────────────────────────────────┤
│                                     │
│  📤 UPLOAD TAB                      │
│  • Drag & drop images               │
│  • Click to select file              │
│  • See instant prediction            │
│                                     │
│  📷 CAMERA TAB                      │
│  • Start camera                      │
│  • Capture photo of fabric           │
│  • Get GSM result                    │
│                                     │
│  📁 BATCH TAB                       │
│  • Select multiple images            │
│  • Process all at once               │
│  • View all predictions              │
│                                     │
└─────────────────────────────────────┘
```

---

## 📚 Documentation Guide

Choose your path:

### 🏃 I just want it working!
→ Read: **QUICKSTART.md** (5 min read)

### 🔧 I want full details
→ Read: **README.md** (20 min read)

### 📋 I want a checklist
→ Read: **SETUP_COMPLETE.md** (10 min read)

### 🗺️ I want to understand it all
→ Read: **INDEX.md** (comprehensive guide)

---

## 🐛 Quick Troubleshooting

### "No module named flask"
```bash
pip install -r requirements.txt
```

### "Model file not found"
Check: Does `../Model/gsm_regressor.pt` exist?
If not: Get the model file from your training

### "Port 5000 already in use"
```bash
# Windows
netstat -ano | findstr :5000

# Mac/Linux
lsof -i :5000
```

### "Camera won't work"
- Check browser permissions
- Try a different browser (Chrome works best)
- Ensure you're on HTTPS (production)

---

## 🎨 What's Included

```
flask_app/
├── 📄 app.py                    Main app (400+ lines)
├── 📄 extract_fabric_features.py  Feature extraction
├── 📄 setup.py                  Setup script
├── 📄 test.py                   Testing suite
├── 📄 requirements.txt           Dependencies
├── 📄 config.json               Configuration
├── 🌐 templates/index.html      Web interface
├── 🎨 static/style.css          Styling
└── 📚 Documentation files:
    ├── QUICKSTART.md            ← Start here
    ├── README.md                ← Full docs
    ├── SETUP_COMPLETE.md        ← Setup guide
    └── INDEX.md                 ← Overview
```

---

## 💡 Common Questions

**Q: What images should I upload?**
A: Fabric microscopy images (512×512 best)
   - PNG, JPG, JPEG, BMP, TIFF
   - Max 16MB per file
   - Clear, well-lit images

**Q: What's the output?**
A: GSM value (weight in grams per square meter)
   - Example: 150.25 g/m²
   - Confidence level: High/Medium

**Q: How long does prediction take?**
A: First: ~3-5 seconds (loading model)
   Rest: ~1-2 seconds per image

**Q: Can I use this on mobile?**
A: Yes! UI is responsive for phones/tablets
   Camera capture works on smartphones

**Q: Can I integrate with my app?**
A: Yes! Use the REST API:
   ```bash
   curl -X POST -F "image=@fabric.jpg" \
     http://localhost:5000/api/predict
   ```

---

## 🚀 Feature List

✅ Web interface (upload/camera/batch)
✅ REST API (single & batch predict)
✅ Real-time camera capture
✅ 64 feature extraction
✅ CatBoost predictions
✅ Confidence scoring
✅ Responsive design
✅ Error handling
✅ Health check endpoint
✅ Setup validation
✅ Testing suite
✅ Comprehensive docs

---

## 📊 Example Predictions

```
Input: Fabric microscopy image (512×512)

Processing:
1. Preprocess image
2. Extract 64 features
3. Scale features
4. Load CatBoost model
5. Make prediction

Output:
{
  "gsm_prediction": 150.25,
  "confidence": "High",
  "timestamp": "2024-01-17T10:30:00"
}

Interpretation:
• Fabric weight: 150.25 g/m²
• Prediction is RELIABLE
• Processed in 1.2 seconds
```

---

## 🔌 API Examples

### Web Interface (Easiest)
1. Open http://localhost:5000
2. Click "Upload Image"
3. Select your fabric image
4. See result instantly

### Command Line (cURL)
```bash
curl -X POST -F "image=@fabric.jpg" \
  http://localhost:5000/api/predict
```

### Python Script
```python
import requests

with open('fabric.jpg', 'rb') as f:
    r = requests.post(
        'http://localhost:5000/api/predict',
        files={'image': f}
    )
    print(r.json())
```

### JavaScript (Web)
```javascript
const formData = new FormData();
formData.append('image', imageFile);

fetch('/api/predict', {
    method: 'POST',
    body: formData
})
.then(r => r.json())
.then(data => console.log(data));
```

---

## 🎯 Next Steps

### Right Now
1. ✅ Run `pip install -r requirements.txt`
2. ✅ Run `python setup.py`
3. ✅ Run `python app.py`
4. ✅ Open http://localhost:5000

### After It Works
1. Test with your images
2. Try batch upload
3. Try camera capture
4. Check API endpoints

### Later
1. Customize styling
2. Add to your website
3. Deploy to cloud
4. Set up monitoring

---

## 📞 Help

**Something not working?**

1. **Check logs:** Look at Flask console output
2. **Run tests:** `python test.py`
3. **Read docs:** Start with README.md
4. **Check setup:** Run `python setup.py` again

**Common issues:**
- Model not found → Check MODEL_PATH in app.py
- Port in use → Change port in app.py
- Dependencies missing → pip install -r requirements.txt
- Camera not working → Check browser permissions

---

## ✨ You're Ready!

Everything is set up and ready to use.

**To start:**
```bash
cd flask_app
python app.py
```

**Then visit:**
🌐 http://localhost:5000

**That's it!** 🎉

Upload a fabric image and see your GSM prediction.

---

## 📖 More Info

**Quick questions:** See "Common Questions" above

**Need full docs:** Read [README.md](README.md)

**Need setup help:** Read [SETUP_COMPLETE.md](SETUP_COMPLETE.md)

**Want overview:** Read [INDEX.md](INDEX.md)

---

**Happy predicting!** 🧵🎯

Created: January 17, 2026
Version: 1.0.0
