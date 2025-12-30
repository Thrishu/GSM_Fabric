# Fabric GSM Prediction Pipeline - Feature Extraction Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [How It Works](#how-it-works)
3. [Feature Extraction Methods](#feature-extraction-methods)
4. [Complete Feature List](#complete-feature-list)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Configuration Guide](#configuration-guide)

---

## System Overview

The **Fabric GSM Prediction Pipeline** is a machine learning system that predicts fabric GSM (grams per square meter) from microscopic fabric images. It combines three complementary feature extraction approaches and trains regression models for accurate predictions.

### Key Components
- **Image Preprocessing**: Standardization and enhancement
- **Texture Feature Extraction**: Low-level statistical descriptors (GLCM, LBP)
- **Deep Learning Features**: High-level semantic features (MobileNet CNN)
- **Feature Fusion**: Combining heterogeneous features with scaling
- **Regression Models**: Random Forest and Gradient Boosting regressors

### Design Philosophy
The system leverages:
- **Low-level features** (GLCM/LBP) to capture fine-grained texture details
- **High-level features** (CNN) to capture semantic patterns learned from ImageNet
- **Hybrid approach** to maximize predictive power while maintaining interpretability

---

## How It Works

### Complete Pipeline Flow

```
Fabric Image
    ↓
[1] IMAGE PREPROCESSING
    ├─ Load image (BGR → Grayscale)
    ├─ Resize to 224×224
    ├─ Apply CLAHE enhancement
    └─ Normalize pixel values [0, 1]
    ↓
[2] PARALLEL FEATURE EXTRACTION
    ├─ TEXTURE FEATURES (Grayscale)
    │  ├─ GLCM (contrast, homogeneity, energy, correlation)
    │  └─ LBP (Local Binary Pattern histogram)
    │
    └─ DEEP FEATURES (RGB or Grayscale)
       └─ MobileNet CNN feature extraction
    ↓
[3] FEATURE FUSION
    ├─ Concatenate texture + deep features
    ├─ Apply StandardScaler or MinMaxScaler
    └─ Create unified feature vector
    ↓
[4] REGRESSION MODEL
    ├─ Random Forest Regressor
    └─ OR Gradient Boosting Regressor
    ↓
GSM Prediction (continuous value)
```

### Execution Timeline

1. **Initialization Phase**
   - Load configuration from `config.yaml`
   - Initialize extractors (TextureFeatureExtractor, DeepFeatureExtractor)
   - Initialize scalers and models

2. **Training Phase**
   - Load and preprocess training images
   - Extract texture features from grayscale images
   - Extract deep features from RGB images
   - Fuse and scale features
   - Train regression model
   - Fit scalers on training data

3. **Inference Phase**
   - Preprocess new image
   - Extract texture and deep features
   - Apply fitted scalers
   - Generate GSM prediction

---

## Feature Extraction Methods

### 1. Image Preprocessing

**Purpose**: Standardize input images and enhance visibility of fabric structures

#### Steps:

**a) Image Loading & Conversion**
- Load image from disk (OpenCV BGR format)
- Convert to grayscale for texture analysis
- Preserve RGB for CNN processing

**b) Resizing**
- Target size: 224×224 pixels (standard for MobileNet)
- Method: Cubic interpolation (high-quality)
- Preserves aspect ratio: Images stretched/compressed to exact size

**c) CLAHE Enhancement** (Contrast Limited Adaptive Histogram Equalization)
- **Why**: Microscopy images often have uneven illumination
- **How**: Divides image into 8×8 tile grid, applies local histogram equalization
- **Benefits**: Enhances local contrast without amplifying noise
- **Parameters**:
  - Clip limit: 2.0 (prevents over-amplification)
  - Tile size: 8×8 (balances local and global processing)

**d) Normalization**
- Maps pixel values to [0, 1] range
- Ensures features have similar magnitude scales
- Required for downstream models

---

### 2. Texture Feature Extraction

Texture features capture the statistical properties of pixel intensity patterns, revealing fabric structure and weave patterns.

#### Method A: GLCM (Gray Level Co-occurrence Matrix)

**Concept**: Analyzes how often pixel pairs with specific intensities occur at specific distances and directions.

**Mathematical Foundation**:
- Count pixel pair frequencies at distance $d$ and angle $\theta$
- Normalize to create probability matrix
- Compute statistics from this matrix

**Parameters**:
```yaml
distances: [1, 3]        # Pixels apart (1 = adjacent, 3 = farther)
angles: [0°, 45°, 90°, 135°]  # Four cardinal directions
levels: 256              # Quantization levels for intensity
metrics: 4 types         # See below
```

**Extracted Metrics** (for each distance × angle combination):

| Metric | Meaning | Use Case |
|--------|---------|----------|
| **Contrast** | Local variation in intensity | Measures fine details, weave visibility |
| **Homogeneity** | Uniformity of the texture | High for smooth fabrics, low for rough ones |
| **Energy** (Angular 2nd Moment) | Orderliness/uniformity | Distinguishes regular vs random patterns |
| **Correlation** | Pixel linear dependencies | Captures long-range structure |

**Feature Count from GLCM**:
- Distances: 2 × Angles: 4 × Metrics: 4 = **32 features**

**Example Interpretation**:
- High contrast + low homogeneity → Tightly woven fabric
- High energy → Regular, predictable weave pattern
- High correlation → Strong directional structure

---

#### Method B: LBP (Local Binary Pattern)

**Concept**: Compares each pixel to its neighbors, creating a binary pattern that captures local texture.

**Process**:
1. For each pixel, compare it to 8 neighbors in a circle
2. Create 8-bit binary code (neighbor ≥ center pixel = 1, else 0)
3. Encode as 0-255 value
4. Create histogram of all such patterns
5. Extract uniform patterns only (patterns with few transitions)

**Parameters**:
```yaml
radius: 3              # Distance from center pixel to neighbors
n_points: 8            # Number of neighbors sampled (8 for radius=3)
method: "uniform"      # Limits to patterns with ≤2 binary transitions
n_bins: 59             # Number of histogram bins (uniform LBP)
```

**Why "Uniform" Patterns?**
- Out of 256 possible 8-bit patterns, only 59 are "uniform"
- Uniform = patterns with ≤2 transitions (e.g., 00011110)
- These capture fundamental texture primitives
- Significantly reduces feature dimensionality

**Feature Count from LBP**:
- **59 features** (one bin per uniform pattern)

**Example Interpretation**:
- Peak at bin 0 or 255 → Very uniform texture (smooth fabric)
- Distributed across bins → Complex texture (rough/patterned)
- Specific pattern prevalence → Characteristic weave structure

---

### 3. Deep Learning Feature Extraction

**Purpose**: Extract high-level semantic features learned from ImageNet, capturing patterns humans don't explicitly define.

#### Architecture: MobileNetV2 / MobileNetV3

**Why MobileNet?**
- Designed for mobile/edge devices (Raspberry Pi compatible)
- Efficient yet accurate (depthwise separable convolutions)
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Fast inference with small memory footprint

**Network Structure**:
```
Input Image (224×224×3)
    ↓
Convolutional Blocks (multiple stages)
├─ Early layers: Edge detection, color patterns
├─ Middle layers: Texture patterns, shapes
└─ Late layers: Object parts, semantic features
    ↓
Global Average Pooling (collapse spatial dimensions)
    ↓
Feature Vector (1D)
```

**Feature Extraction Process**:
1. Remove classification head (final softmax layer)
2. Pass image through convolutional layers
3. Apply global average pooling to convert 7×7×1280 → 1×1280
4. Output: **1280-dimensional feature vector** (MobileNetV2)

**What These Features Represent**:
- **Layer 1-2**: Raw edges (horizontal, vertical, diagonal)
- **Layer 3-5**: Textures and simple patterns
- **Layer 6-10**: Complex textures, shapes, object parts
- **Layer 11-13**: Semantic concepts, fabric properties

**Transfer Learning Advantage**:
- Network trained on millions of images
- Learned representations transfer well to fabric analysis
- No need to train from scratch
- Requires minimal data for fine-tuning

**Feature Count from Deep Learning**:
- **1280 features** (MobileNetV2)
- Can use MobileNetV3Small (576) or MobileNetV3Large (1280) for speed/accuracy trade-off

---

### 4. Feature Fusion

**Purpose**: Combine diverse features into a single, standardized representation.

#### Fusion Strategy

```
Texture Features (91 dims)     Deep Features (1280 dims)
        ↓                                ↓
        └────────────────┬───────────────┘
                         ↓
              Concatenation (1371 dims)
                         ↓
              StandardScaler (z-score)
              or MinMaxScaler ([0,1])
                         ↓
              Final Feature Vector (1371 dims)
```

**Why Scaling?**
- GLCM metrics typically range: [0, 1000+]
- LBP histogram: [0, image_area]
- Deep CNN features: [−5, +5] (normalized activations)
- **Problem**: Disparate ranges cause imbalance in regression models
- **Solution**: Standardize all features to mean=0, std=1 (StandardScaler)

**Formula** (StandardScaler):
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

where $\mu$ is feature mean and $\sigma$ is standard deviation.

**Scaler Persistence**:
- Fit scaler on training data only
- Save scaler object to disk
- Apply same scaler transformation during inference
- Ensures consistency between training and prediction

---

## Complete Feature List

### Summary Table

| Feature Group | Method | Count | Dimensionality | Range |
|---------------|--------|-------|-----------------|-------|
| GLCM | Co-occurrence matrix | 32 | Fixed (2×4×4) | [0, 1] |
| LBP | Local binary pattern | 59 | Histogram bins | [0, image_area] |
| **Texture Total** | Combined | **91** | - | - |
| **Deep Features** | MobileNetV2 CNN | 1280 | Fixed | [−5, +5] |
| **Fused Total** | Concatenated + scaled | **1371** | - | [−3, +3] (after StandardScaler) |

---

### Detailed Feature Breakdown

#### GLCM Features (32 total)

**Structure**: For each (distance, angle) pair → extract 4 metrics

```
Distances: [1, 3]
Angles: [0°, 45°, 90°, 135°]
Metrics: [contrast, homogeneity, energy, correlation]

Total combinations: 2 × 4 × 4 = 32 features
```

**Feature Naming Convention**:
- `glcm_contrast_d1_0deg`, `glcm_contrast_d1_45deg`, ...
- `glcm_homogeneity_d3_135deg`, ...
- etc.

**Physical Interpretation**:

| Feature | Low Value | High Value |
|---------|-----------|------------|
| Contrast | Smooth, uniform texture | Rough, variable texture |
| Homogeneity | Mixed textures | Uniform texture |
| Energy | Random, chaotic patterns | Ordered, regular patterns |
| Correlation | Weak structure | Strong directional structure |

---

#### LBP Features (59 total)

**Structure**: Histogram of 59 uniform binary patterns

```
Pattern Examples:
- 00000000 (all neighbors dark) → Bin 0
- 00001000 (one transition) → Uniform bin
- 00110011 (alternating) → Not uniform, ignored
- 11111111 (all neighbors bright) → Bin 58

Histogram: [count_pattern_0, count_pattern_1, ..., count_pattern_58]
Total: 59 features
```

**Interpretation**:
- Values normalized to histogram (relative frequencies)
- Shows distribution of micro-patterns
- Captures texture "fingerprint"

---

#### Deep Features (1280 total)

**From MobileNetV2 activation maps:**

```
Layer Name: global_average_pooling2d
Input: [7, 7, 1280]  (spatial feature maps)
Operation: Average pool across all 49 spatial locations
Output: [1280]        (single vector)
```

**Semantic Information**:
- Early dimensions: Low-level edge/corner responses
- Middle dimensions: Texture and pattern activations
- Late dimensions: High-level semantic concepts
- Interpretability: Lower than GLCM/LBP (black-box CNN)

---

## Pipeline Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                   FABRIC GSM PREDICTION PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼────────┐   ┌──────▼──────────┐
            │ CONFIG MANAGER │   │ IMAGE PROCESSOR │
            │ (config.yaml)  │   │ (StandardInput) │
            └────────────────┘   └──────┬──────────┘
                    │                    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼─────────┐
                    │ DATASET LOADER    │
                    │ (FabricNet)       │
                    └─────────┬─────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    │          ┌──────────────▼────────────────┐        │
    │          │ IMAGE PREPROCESSING           │        │
    │          │ ├─ Grayscale conversion       │        │
    │          │ ├─ Resizing (224×224)         │        │
    │          │ ├─ CLAHE enhancement          │        │
    │          │ └─ Normalization ([0,1])      │        │
    │          └──────────────┬────────────────┘        │
    │                         │                         │
    ├─────────────────────────┼──────────────────────────┤
    │        ┌────────────────▼──────────────┐           │
    │        │ TEXTURE FEATURES               │           │
    │        │ TextureFeatureExtractor        │           │
    │        ├─ GLCM (32 features)           │           │
    │        └─ LBP (59 features)            │           │
    │             = 91 features total        │           │
    │        └─────────┬──────────────────────┘           │
    │                  │                                 │
    │    ┌─────────────▼──────────────┐                  │
    │    │ DEEP FEATURES              │                  │
    │    │ DeepFeatureExtractor       │                  │
    │    ├─ MobileNetV2               │                  │
    │    │   └─ Global Avg Pool       │                  │
    │    └─ 1280 features             │                  │
    │    └─────────┬──────────────────┘                  │
    │              │                                    │
    └──────────────┼────────────────────────────────────┘
                   │
         ┌─────────▼────────────┐
         │ FEATURE FUSION       │
         │ ├─ Concatenate       │
         │ │  (91 + 1280 = 1371)│
         │ └─ Scale/Normalize   │
         │    (StandardScaler)  │
         └─────────┬────────────┘
                   │
    ┌──────────────▼──────────────┐
    │ TRAIN/VAL/TEST SPLIT       │
    │ (70% / 15% / 15%)          │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │ REGRESSION MODEL TRAINING           │
    │ ├─ Random Forest Regressor          │
    │ └─ Gradient Boosting Regressor      │
    │ (fit on training set)               │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │ MODEL EVALUATION                    │
    │ ├─ MAE (Mean Absolute Error)        │
    │ ├─ RMSE (Root Mean Squared Error)   │
    │ └─ R² Score                         │
    │ (evaluate on test set)              │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │ INFERENCE & PREDICTIONS             │
    │ └─ Output: GSM (g/m²)               │
    └─────────────────────────────────────┘
```

### Module Dependencies

```
fabric_gsm_pipeline/
│
├── configs/
│   └── config.yaml ◄─────────────────────────── Central configuration
│
├── src/
│   ├── utils/
│   │   ├── config.py ─────────┐
│   │   └── logger.py          │
│   │                           │
│   ├── preprocessing/          │
│   │   └── image_preprocessor.py ◄─ Uses: config, logger
│   │                           │
│   ├── features/               │
│   │   ├── texture_features.py ◄─ Uses: logger
│   │   ├── deep_features.py ◄─ Uses: logger, preprocessing
│   │   └── feature_fusion.py ◄─ Uses: logger
│   │                           │
│   ├── models/                 │
│   │   └── regression.py ◄─ Uses: logger
│   │
│   └── data/
│       └── fabricnet_loader.py ◄─ Uses: config
│
└── main.py ◄─ Orchestrates entire pipeline

```

---

## Configuration Guide

### config.yaml Structure

```yaml
data:
  fabricnet_root: "data/FabricNet"    # Dataset location
  image_size: [224, 224]              # Preprocessing target
  train_ratio: 0.7                    # Training split %
  val_ratio: 0.15                     # Validation split %
  test_ratio: 0.15                    # Test split %
  random_seed: 42                     # Reproducibility

preprocessing:
  enable_clahe: true                  # Contrast enhancement
  clahe_clip_limit: 2.0               # CLAHE intensity
  clahe_tile_size: [8, 8]             # Grid granularity
  normalization: "minmax"             # Scaling method

texture_features:
  glcm:
    enable: true
    distances: [1, 3]                 # Pixel distances
    angles: [0, 45, 90, 135]          # Directions
    levels: 256                       # Quantization
    metrics: [contrast, homogeneity, energy, correlation]
  
  lbp:
    enable: true
    radius: 3                         # Neighborhood size
    n_points: 8                       # Sampling points
    method: "uniform"                 # Pattern type
    n_bins: 59                        # Histogram bins

deep_features:
  model_type: "MobileNetV2"           # Architecture choice
  weights: "imagenet"                 # Pretrained weights
  pooling: "global_average"           # Aggregation method
  use_gpu: true                       # Hardware acceleration

regression:
  model_type: "random_forest"         # or "gradient_boosting"
  random_forest:
    n_estimators: 100                 # Number of trees
    max_depth: 20                     # Tree depth limit
    min_samples_split: 5              # Split threshold
    random_state: 42

  gradient_boosting:
    n_estimators: 100                 # Boosting rounds
    learning_rate: 0.1                # Step size
    max_depth: 5                      # Tree depth
    random_state: 42
```

### Key Parameters to Adjust

| Parameter | Effect | When to Change |
|-----------|--------|-----------------|
| `clahe_clip_limit` | Contrast enhancement strength | If images too bright/dark |
| `glcm_distances` | Texture analysis scale | Larger for coarse weaves |
| `lbp_radius` | Local pattern size | Larger for larger features |
| `n_estimators` | Model ensemble size | More for complex patterns |
| `learning_rate` | Boosting step size | Smaller for stability |

---

## Practical Example

### Scenario: Predicting GSM for a Cotton Fabric

**Input**: Microscopy image of cotton fabric weave

**Step 1: Preprocessing**
```
Raw image (2048×2048, varying brightness)
  → Grayscale conversion
  → Resize to 224×224
  → CLAHE enhancement (increases local contrast)
  → Normalize to [0, 1]
Result: Standardized image ready for analysis
```

**Step 2: Texture Feature Extraction**
```
GLCM Analysis (grayscale image):
  - At distance=1, angle=0°: contrast=45.2, homogeneity=0.78, ...
  - At distance=1, angle=45°: contrast=52.1, homogeneity=0.72, ...
  ... (continue for all 8 distance×angle combinations)
  
LBP Analysis:
  - Uniform pattern 00000001: 234 occurrences
  - Uniform pattern 00001111: 567 occurrences
  ... (histogram of 59 patterns)
  
Result: 91-dimensional texture feature vector
```

**Step 3: Deep Feature Extraction**
```
MobileNetV2 CNN:
  - Input: 224×224×3 RGB image
  - Forward pass through 150+ convolutional layers
  - Global average pooling: 7×7×1280 → 1280
  
Result: 1280-dimensional CNN feature vector
```

**Step 4: Feature Fusion**
```
Concatenate: [91 texture features] + [1280 CNN features]
           = 1371-dimensional vector

StandardScaler Transform:
  - Each feature: (value - training_mean) / training_std
  - Result: Mean ≈ 0, Std ≈ 1 for all features
```

**Step 5: Regression Prediction**
```
Random Forest Model (trained on 1371-dim features):
  - 100 decision trees
  - Each tree votes on GSM value
  - Average 100 votes
  
Result: GSM ≈ 175 g/m² (final prediction)
```

---

## Summary

This pipeline extracts **1,371 features** from each fabric image:

1. **91 Texture Features** - Capture fine-grained weave structure
2. **1280 Deep Features** - Capture high-level semantic patterns

Features are:
- **Extracted in parallel** (efficiency)
- **Fused systematically** (combining perspectives)
- **Scaled uniformly** (fair contribution)
- **Used by regression models** (predicting GSM)

The hybrid approach balances **interpretability** (texture features) with **predictive power** (deep learning), making it suitable for both research and production applications.
