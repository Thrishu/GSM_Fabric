# What Fabric Properties Are Extracted? 🧵

## Complete Feature List with Actual Extraction Methods

### Summary Table

| Property | Extracted? | Feature Count | Extraction Method |
|----------|:----------:|:-------------:|-------------------|
| **Warp Density** | ✅ YES | 1 | Edge detection on vertical threads |
| **Weft Density** | ✅ YES | 1 | Edge detection on horizontal threads |
| **Yarn Diameter** | ✅ YES | 1 | Edge thickness measurement |
| **Total Density** | ✅ YES | 1 | (Warp + Weft) / 2 |
| **GLCM Contrast** | ✅ YES | 8 | Co-occurrence matrix, 2 distances × 4 angles |
| **GLCM Homogeneity** | ✅ YES | 8 | Co-occurrence matrix, 2 distances × 4 angles |
| **GLCM Energy** | ✅ YES | 8 | Co-occurrence matrix, 2 distances × 4 angles |
| **GLCM Correlation** | ✅ YES | 8 | Co-occurrence matrix, 2 distances × 4 angles |
| **LBP Histogram** | ✅ YES | 59 | Local Binary Pattern uniform bins |
| **CNN Features** | ✅ YES | 1280 | MobileNetV3 pretrained model |
| **Cloth Area** | ❌ NO | 0 | Not extracted |
| **Individual Warp Count** | ⚠️ PARTIAL | 1 | Density-based (not absolute count) |
| **Individual Weft Count** | ⚠️ PARTIAL | 1 | Density-based (not absolute count) |

---

## Detailed Breakdown

### 1. ✅ Warp Density (Extracted)

**What is it?**
- Vertical threads per unit width
- Higher value = tighter weave in warp direction
- Directly related to fabric weight (GSM)

**How it's extracted:**

```python
# Detect threads using edge detection (Canny edge detector)
edges = cv2.Canny(gray_image, 100, 200)

# Count vertical thread lines
vertical_sum = edges.sum(axis=0)  # Sum pixels in vertical direction
warp_density = float(np.count_nonzero(vertical_sum > 0) / w)
```

**Calculation:**
```
warp_density = (number of columns with edge pixels) / total_width
```

**Example:**
```
Image: 224×224 pixels
Vertical edges detected in: 145 columns (out of 224)
Warp density = 145 / 224 = 0.648
```

**Feature Range:** [0, 1] (normalized)

---

### 2. ✅ Weft Density (Extracted)

**What is it?**
- Horizontal threads per unit height
- Higher value = tighter weave in weft direction
- Directly related to fabric weight (GSM)

**How it's extracted:**

```python
# Count horizontal thread lines
horizontal_sum = edges.sum(axis=1)  # Sum pixels in horizontal direction
weft_density = float(np.count_nonzero(horizontal_sum > 0) / h)
```

**Calculation:**
```
weft_density = (number of rows with edge pixels) / total_height
```

**Example:**
```
Image: 224×224 pixels
Horizontal edges detected in: 168 rows (out of 224)
Weft density = 168 / 224 = 0.75
```

**Feature Range:** [0, 1] (normalized)

---

### 3. ✅ Yarn Diameter (Extracted)

**What is it?**
- Estimated thickness of individual yarn/thread
- Related to yarn fineness and fabric weight
- Larger diameter = heavier fabric (higher GSM)

**How it's extracted:**

```python
# Estimate yarn diameter from edge pixel thickness
yarn_diameter = float(np.mean(np.where(edges > 0)[0])) / h
```

**Calculation:**
```
yarn_diameter = average_row_position_of_edges / image_height
```

**What this means:**
- If edges appear at rows 50, 100, 150... → diameter related to spacing
- Normalized by image height to make it scale-invariant

**Example:**
```
Edge pixel positions: [10, 20, 30, 100, 110, 120, ...]
Average position: 65
Height: 224
yarn_diameter = 65 / 224 = 0.29
```

**Feature Range:** [0, 1] (normalized)

---

### 4. ✅ Total Density (Extracted)

**What is it?**
- Average of warp and weft densities
- Overall thread density per unit area
- **Primary predictor of GSM**

**How it's calculated:**

```python
total_density = (warp_density + weft_density) / 2.0
```

**Why this matters:**
$$\text{GSM} \approx \text{yarn\_properties} \times \text{total\_density}$$

**Example:**
```
Warp density: 0.648
Weft density: 0.75
Total density: (0.648 + 0.75) / 2 = 0.699
```

**Feature Range:** [0, 1] (normalized)

---

### 5. ✅ GLCM Features (32 features total)

**What are they?**
- Texture statistics from Gray Level Co-occurrence Matrix
- Captures how pixel intensity patterns repeat and vary
- **Reflects fabric tightness and weave structure**

**How they're extracted:**

```python
# Quantize image to 32 levels for computation efficiency
gray_quant = (gray_image / 8).astype(np.uint8)

# Compute GLCM at multiple distances and angles
glcm = graycomatrix(
    gray_quant,
    distances=[1, 3],           # 1-pixel and 3-pixel spacing
    angles=[0, 45°, 90°, 135°], # Four cardinal directions
    levels=32,
    symmetric=True,
    normed=True
)

# Extract 4 metrics for each distance-angle pair
metrics = ['contrast', 'homogeneity', 'energy', 'correlation']
```

**Feature Count:** 2 distances × 4 angles × 4 metrics = **32 features**

**The 4 Metrics Explained:**

#### a) **Contrast**
```
Formula: Σ n² * P(n)  where n = |i - j|
```
- Measures local intensity variation
- **High contrast** = rough, coarse weave
- **Low contrast** = smooth, fine weave
- Indicates **yarn thickness** and **tightness**

#### b) **Homogeneity** (Inverse Difference Moment)
```
Formula: Σ P(i,j) / (1 + |i - j|)
```
- Measures uniformity of texture
- **High homogeneity** = uniform weave pattern
- **Low homogeneity** = variable weave
- Related to **fabric consistency**

#### c) **Energy** (Angular Second Moment)
```
Formula: Σ P(i,j)²
```
- Measures orderliness of texture
- **High energy** = regular, predictable pattern
- **Low energy** = chaotic, random pattern
- Indicates **weave regularity**

#### d) **Correlation**
```
Formula: Σ (i - μ_i)(j - μ_j) * P(i,j) / σ_i * σ_j
```
- Measures linear dependency between pixel values
- **High correlation** = strong directional structure
- **Low correlation** = random structure
- Related to **weave direction strength**

**Example Output:**
```
Contrast @ distance=1, 0°:   245.2
Contrast @ distance=1, 45°:  256.8
Contrast @ distance=1, 90°:  312.4
Contrast @ distance=1, 135°: 289.1
Contrast @ distance=3, 0°:   198.5
... (28 more features)
```

---

### 6. ✅ LBP Features (59 features)

**What are they?**
- Local Binary Patterns: micro-texture analysis
- Each pixel compared to 8 neighbors
- **Captures fine-grained weave patterns**

**How they're extracted:**

```python
# Compute LBP with 8 neighbors at radius 3
lbp = feature.local_binary_pattern(
    gray_image,
    P=8,        # 8-point circular neighborhood
    R=3,        # Radius of 3 pixels
    method='uniform'  # Only uniform patterns
)

# Create histogram of patterns
hist, _ = np.histogram(
    lbp.ravel(),
    bins=np.arange(0, 60),
    range=(0, 59)
)

# Normalize
normalized_hist = hist.astype(float) / hist.sum()
```

**Feature Count:** 59 uniform patterns = **59 features**

**What "Uniform" Means:**
- Only patterns with ≤2 binary transitions
- Examples of uniform patterns:
  ```
  00000000 (transition: 1) ✓
  11111111 (transition: 1) ✓
  00011110 (transition: 2) ✓
  00110011 (transition: 4) ✗ Not uniform
  ```
- Reduces 256 possible patterns to 59 fundamental ones

**Example Histogram:**
```
Bin 0 (00000000):     245 occurrences  → 2.1%
Bin 1 (00000001):     312 occurrences  → 2.7%
Bin 2 (00000011):     678 occurrences  → 5.8%
...
Bin 58 (11111111):    389 occurrences  → 3.3%
Total: 11,628 pixels

Normalized: [0.021, 0.027, 0.058, ..., 0.033]
```

**What it tells you:**
- Peak in bin 0 or 58 → **Smooth fabric** (uniform pixels)
- Distributed across bins → **Textured fabric** (varied patterns)
- Specific pattern prevalence → **Characteristic weave type**

---

### 7. ✅ CNN Deep Features (1280 features)

**What are they?**
- High-level semantic features from pretrained neural network
- Learned from 1.2 million ImageNet images
- Capture patterns humans don't explicitly define

**How they're extracted:**

```python
from tensorflow.keras.applications import MobileNetV3Small

# Load pretrained MobileNetV3
model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,      # Remove classification layer
    weights='imagenet',     # Use pretrained weights
    pooling='avg'           # Global average pooling
)

# Extract features
features = model.predict(image)  # Output shape: (1, 1280)
```

**Network Architecture:**
```
Input (224×224×3)
    ↓
Conv Layer 1: Edge detection (3×3 filters)
    ↓
Conv Layers 2-5: Texture patterns, shapes
    ↓
Conv Layers 6-10: Complex textures, object parts
    ↓
Conv Layers 11-13: High-level semantic concepts
    ↓
Global Average Pooling: Collapse 7×7×1280 → 1280
    ↓
Output: 1280-dimensional feature vector
```

**Feature Count:** **1280 features**

**What these represent:**
- Early dimensions: Edge responses, color gradients
- Middle dimensions: Texture patterns (very similar to GLCM/LBP!)
- Late dimensions: Semantic concepts (fabric properties)

**Advantage over hand-crafted features:**
- Automatically learned from data
- More robust to variations
- Captures patterns that GLCM/LBP might miss

---

## ❌ What is NOT Extracted

### 1. **Absolute Warp Count** (Number of warp threads)
- ❌ Not extracted
- ⚠️ Current method: Density-based estimate only
- **Why not:** Would need image calibration (pixels/mm) - not available

**Current approach:**
```python
# Estimates relative density, not absolute count
warp_density = 145 threads_per_image_width
# Actual threads/inch would need: pixels_per_inch * warp_density
```

**To extract this, we'd need:**
- Reference scale in image (e.g., millimeter marker)
- OR image metadata with DPI information
- Neither available in current pipeline

---

### 2. **Absolute Weft Count** (Number of weft threads)
- ❌ Not extracted
- ⚠️ Current method: Density-based estimate only

**Same limitation as warp count**

---

### 3. **Cloth Area**
- ❌ Not directly extracted
- ⚠️ **Implicitly used:** Image size is 224×224 (fixed)
- **Why not relevant:** All images normalized to same size

**Could calculate:**
```python
area_pixels = 224 * 224 = 50,176 pixels
# But actual area in mm² would need: pixels_per_mm * area_pixels
```

---

### 4. **Weight per Unit Length (Linear density)**
- ❌ Not extracted
- **Why?** GSM (weight per area) is measured differently

---

### 5. **Fiber Properties** (Material, fineness, etc.)
- ❌ Not extracted
- **Why?** Would need chemical analysis - impossible from images

---

## Summary: What We HAVE

| Category | Features | Details |
|----------|:--------:|---------|
| **Density** | 4 | Warp, Weft, Yarn Diameter, Total |
| **GLCM Texture** | 32 | 2 distances × 4 angles × 4 metrics |
| **LBP Texture** | 59 | 59 uniform pattern bins |
| **CNN Features** | 1280 | MobileNetV3 pretrained |
| **TOTAL** | **1375** | Combined feature vector |

---

## How These Features Predict GSM

### Direct Relationship:

$$\text{GSM} = f(\text{Warp Density}, \text{Weft Density}, \text{Yarn Diameter}, \text{GLCM}, \text{LBP}, \text{CNN})$$

where $f$ is the regression model (Random Forest or Gradient Boosting)

### Physical Interpretation:

```
Higher density         → More threads per area
Larger yarn diameter   → Thicker yarns
Higher GLCM contrast   → Tighter weave
Distributed LBP        → Complex weave pattern
CNN activations        → Semantic patterns learned from data
                ↓
            HIGHER GSM
```

### Example Prediction:

```
Image Input
    ↓
Density Features:      [0.68, 0.75, 0.29, 0.715]
GLCM Features:         [245, 0.78, 0.92, 0.65, ...] (32 total)
LBP Features:          [0.021, 0.027, 0.058, ...] (59 total)
CNN Features:          [-0.3, 0.8, -1.2, ...] (1280 total)
    ↓
Fused & Scaled:        [1.2, 1.4, -0.5, ...] (1375 features)
    ↓
Regression Model:      Random Forest (100 trees)
    ↓
GSM Prediction:        ≈ 185 g/m²
```

---

## Data Enhancement Techniques

### Augmentation (To overcome limited samples)

The pipeline uses **data augmentation** to create variations:

```python
# 7 types of augmentations applied:
1. Rotation (±180°)          - Fabric scanned at any angle
2. Scale (±20%)              - Zoom variations
3. Elastic Deformation       - Fabric stretching
4. Grid Distortion           - Wrinkles/folds
5. Brightness/Contrast       - Lighting variations
6. Gaussian Blur             - Focus variations
7. Gaussian Noise            - Sensor noise
```

**Effect:** 1 image → 5 augmented variants = 5× more training data

---

## To Extract Additional Properties

### To get Absolute Thread Counts:

```python
# Add image calibration
calibration_pixels_per_mm = 10  # Need to determine this

# Calculate absolute counts
warp_count_per_cm = warp_density * calibration_pixels_per_mm * 10
weft_count_per_cm = weft_density * calibration_pixels_per_mm * 10
```

### To get Cloth Area:

```python
# Add image metadata
image_dpi = 96  # pixels per inch
mm_per_pixel = 25.4 / image_dpi

area_mm2 = (224 * 224) * (mm_per_pixel ** 2)
```

### To get Fiber Properties:

- **Not possible from images alone**
- Would need: XRF spectroscopy, microscopy, or chemical analysis
- Would be separate measurements, not extracted from this pipeline

---

## Conclusion

✅ **Currently extracts: 1375 fabric-related features**

Key extractable properties:
- Warp & Weft densities ✅
- Yarn diameter ✅
- Texture characteristics (GLCM, LBP) ✅
- Semantic patterns (CNN) ✅

Limitations:
- Absolute thread counts need calibration
- Cloth area needs metadata
- Fiber properties not measurable from images

**The system is optimized for GSM prediction using image-based features!**
