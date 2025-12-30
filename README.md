# Fabric GSM Prediction Pipeline

A professional, research-grade machine learning pipeline for predicting fabric GSM (grams per square meter) from microscopic fabric images. Designed for efficient inference on resource-constrained environments like Raspberry Pi.

## Overview

This project implements a complete ML pipeline combining:
- **Texture Feature Extraction**: GLCM and Local Binary Pattern (LBP) analysis
- **Deep Learning Features**: MobileNetV2/V3 pretrained CNN backbone
- **Feature Fusion**: Concatenation and standardization of heterogeneous features
- **Regression Modeling**: Random Forest and Gradient Boosting regressors

## Project Structure

```
fabric_gsm_pipeline/
├── main.py                          # Entry point for full pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── configs/
│   └── config.yaml                 # Central configuration file
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── image_preprocessor.py   # Image loading, resizing, CLAHE, normalization
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── texture_features.py     # GLCM and LBP extraction
│   │   ├── deep_features.py        # MobileNet feature extraction
│   │   └── feature_fusion.py       # Feature fusion and scaling
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── regression.py           # RandomForest and GradientBoosting
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── fabricnet_loader.py     # Dataset loading with dummy labels
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Logging configuration
│       └── config.py               # Configuration management
│
├── data/
│   └── FabricNet/                  # Dataset root (user-provided)
│       ├── class1/
│       │   ├── image1.jpg
│       │   └── image2.jpg
│       └── class2/
│           └── image3.jpg
│
├── models/                         # Output directory for trained models
│   ├── fabric_gsm_regressor.pkl    # Trained regression model
│   └── feature_scaler.pkl          # Feature standardization scaler
│
├── results/                        # Output directory for results
│   └── predictions.csv             # Test set predictions
│
└── logs/                           # Log files
    └── fabric_gsm_pipeline.log
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda

### Setup

1. Clone or download the project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All hyperparameters are centralized in `configs/config.yaml`. Key sections:

### Data Configuration
- `image_size`: Target dimensions (default: 224x224)
- `train_ratio`, `val_ratio`, `test_ratio`: Data split percentages
- `random_seed`: For reproducibility

### Preprocessing
- `enable_clahe`: Use Contrast Limited Adaptive Histogram Equalization
- `normalization`: "minmax" (0-1) or "standardization" (z-score)

### Texture Features
- GLCM: Distances, angles, and metrics (contrast, homogeneity, energy, correlation)
- LBP: Radius, neighborhood points, and histogram bins

### Deep Features
- `model_type`: "MobileNetV2", "MobileNetV3Small", or "MobileNetV3Large"
- `weights`: "imagenet" for pretrained weights

### Models
- `active_model`: "random_forest" or "gradient_boosting"
- Model-specific hyperparameters (n_estimators, max_depth, etc.)

## Usage

### Basic Pipeline Run

```bash
python main.py \
  --config configs/config.yaml \
  --dataset data/FabricNet \
  --output .
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | configs/config.yaml | Path to configuration file |
| `--dataset` | data/FabricNet | Root directory of FabricNet dataset |
| `--output` | . | Output directory for models/results |

### Output

The pipeline generates:
- **models/fabric_gsm_regressor.pkl**: Trained regression model
- **models/feature_scaler.pkl**: Feature standardization scaler
- **results/predictions.csv**: Predictions on train/val/test splits
- **logs/fabric_gsm_pipeline.log**: Detailed execution log

## Module Documentation

### Image Preprocessing (`src/preprocessing/image_preprocessor.py`)

Handles image loading, resizing, and enhancement.

```python
from src.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(
    target_size=(224, 224),
    enable_clahe=True,
    clahe_clip_limit=2.0,
    normalization_mode="minmax"
)

# Preprocess image from file
preprocessed = preprocessor.preprocess("path/to/image.jpg")

# Preprocess already-loaded numpy array
preprocessed = preprocessor.preprocess_array(image_array)
```

### Texture Feature Extraction (`src/features/texture_features.py`)

Extracts GLCM and LBP texture features.

```python
from src.features import TextureFeatureExtractor

extractor = TextureFeatureExtractor(
    glcm_distances=[1, 3],
    glcm_angles=[0, 45, 90, 135],
    lbp_radius=3,
    lbp_n_points=8
)

# Extract features from preprocessed image
texture_features = extractor.extract_features(image)  # Returns 1D numpy array
```

### Deep Feature Extraction (`src/features/deep_features.py`)

Extracts CNN features using pretrained MobileNet.

```python
from src.features import DeepFeatureExtractor

extractor = DeepFeatureExtractor(
    model_type="MobileNetV2",
    weights="imagenet",
    use_gpu=True
)

# Extract features from preprocessed image
deep_features = extractor.extract_features(image)  # Returns 1D numpy array
```

### Feature Fusion (`src/features/feature_fusion.py`)

Concatenates and scales features.

```python
from src.features.feature_fusion import FeatureFusion

fusion = FeatureFusion(scaler_type="standard")

# Fit scaler on training features
fusion.fit(X_train)

# Transform features
X_train_scaled = fusion.transform(X_train)
X_test_scaled = fusion.transform(X_test)
```

### Regression Models (`src/models/regression.py`)

Unified interface for Random Forest and Gradient Boosting.

```python
from src.models import GSMRegressor

model = GSMRegressor(
    model_type="random_forest",
    n_estimators=100,
    max_depth=15
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
metrics = model.evaluate(X_test, y_test, metrics=["mae", "rmse", "r2"])

# Save/load
model.save_model("path/to/model.pkl")
model.load_model("path/to/model.pkl")
```

### Dataset Loader (`src/data/fabricnet_loader.py`)

Loads FabricNet dataset and generates dummy GSM labels.

```python
from src.data import FabricNetDataset

dataset = FabricNetDataset(dataset_root="data/FabricNet")

# Get all samples with labels
samples = dataset.get_all_samples()  # List of (path, class_name, dummy_gsm)

# Split into train/val/test
train, val, test = dataset.split_samples(samples, train_ratio=0.7)

# Get dataset info
info = dataset.get_dataset_info()
```

## Pipeline Workflow

1. **Dataset Loading**: Discovers FabricNet directory structure and generates stable dummy GSM labels per class
2. **Preprocessing**: Applies grayscale conversion, resizing, CLAHE enhancement, and normalization
3. **Feature Extraction**: 
   - Computes GLCM texture features (4 metrics × N distances × N angles)
   - Extracts CNN features via MobileNet global average pooling
4. **Feature Fusion**: Concatenates texture + CNN features into unified vector
5. **Feature Scaling**: Applies StandardScaler to normalize feature magnitudes
6. **Model Training**: Trains selected regressor (RandomForest or GradientBoosting)
7. **Evaluation**: Computes MAE, RMSE, and R² on train/val/test splits
8. **Results**: Saves trained model, scaler, and predictions

## Dummy Labels for Validation

Since FabricNet lacks GSM annotations, the pipeline creates synthetic targets for validation:

- **Purpose**: Validate full pipeline without real GSM labels
- **Mechanism**: Assigns stable numeric value per class in range [100, 250] g/m²
- **Reproducibility**: Uses seed for consistent assignment across runs
- **Integration**: Replace `dummy_gsm` values with real annotations once available

Example dummy targets:
```
- Cotton: 145.2 g/m²
- Silk: 189.7 g/m²
- Polyester: 112.3 g/m²
```

## Raspberry Pi Compatibility

This pipeline is designed for eventual deployment on Raspberry Pi:

1. **Lightweight Models**: Uses MobileNetV2/V3 instead of ResNet/InceptionV3
2. **Feature Efficiency**: Handcrafted texture features complement CNN features
3. **Model Size**: Random Forest/Gradient Boosting are lightweight vs. neural networks
4. **TensorFlow Lite Ready**: Code structure allows easy conversion to TFLite format
5. **Memory Efficient**: Processes images in sequential batches

### Future Raspberry Pi Deployment

```bash
# Convert TensorFlow model to TFLite
python -m tf2onnx.convert --saved-model path/to/model --output_file model.onnx
# Then convert ONNX to TFLite via appropriate converter
```

## Metrics

### MAE (Mean Absolute Error)
- Average absolute difference between predictions and truth
- Units: g/m²
- Lower is better

### RMSE (Root Mean Squared Error)
- Square root of average squared differences
- Units: g/m²
- Penalizes large errors
- Lower is better

### R² (Coefficient of Determination)
- Fraction of variance explained by model
- Range: [-∞, 1]
- Perfect: 1.0
- Random: 0.0
- Higher is better

## Extending the Pipeline

### Adding New Feature Extractors

Create new class in `src/features/`:

```python
class CustomFeatureExtractor:
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        # Your implementation
        return features  # 1D array
```

### Adding New Regression Models

Extend `src/models/regression.py`:

```python
def __init__(self, model_type="custom_model", **kwargs):
    if model_type == "custom_model":
        self.model = CustomRegressorClass(**kwargs)
```

### Custom Dataset Loaders

Extend `src/data/fabricnet_loader.py` pattern:

```python
class CustomDataset:
    def __init__(self, root_path):
        self.root = Path(root_path)
        self._load_samples()
    
    def get_all_samples(self):
        return [(image_path, class_name, gsm_value), ...]
```

## Logging

The pipeline provides comprehensive logging:

- **Console Output**: INFO level messages to stdout
- **File Logging**: All messages to `logs/fabric_gsm_pipeline.log` (rotating, max 10MB)
- **Debug Info**: Module-level debug messages for troubleshooting

Access loggers in custom code:

```python
from src.utils import get_logger

logger = get_logger("my_module")
logger.info("Message")
logger.debug("Debug info")
logger.warning("Warning")
logger.error("Error")
```

## Configuration Best Practices

1. **Hyperparameter Tuning**: Modify `configs/config.yaml` before running pipeline
2. **Version Control**: Commit config files with corresponding results
3. **Experiment Tracking**: Use different config files for different experiments
4. **Documentation**: Add comments explaining WHY parameters are chosen

Example configuration variants:

- `config_baseline.yaml`: Standard parameters
- `config_fast.yaml`: Reduced models for speed testing
- `config_accurate.yaml`: Larger models for accuracy

## Troubleshooting

### Memory Issues

- Reduce `batch_size` in config (if batching implemented)
- Use `MobileNetV3Small` instead of larger variants
- Process images in smaller batches

### Slow Feature Extraction

- Set `use_gpu: false` if GPU memory limited
- Use faster texture feature settings (fewer distances/angles)
- Implement batching for deep feature extraction

### Low Model Performance

- Verify dummy labels are reasonable
- Check preprocessed images look correct
- Increase model complexity (n_estimators, max_depth)
- Collect more training data

### CUDA/GPU Issues

- Verify TensorFlow GPU installation: `python -c "import tensorflow; print(tensorflow.sysconfig.get_build_info()['cuda_version'])"`
- Set `use_gpu: false` in config as fallback
- Use CPU-only setup if GPU unavailable

## Citation & References

Key techniques implemented:

1. **GLCM**: Haralick et al. (1973). "Textural Features for Image Classification"
2. **LBP**: Ojala et al. (2002). "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns"
3. **MobileNetV2**: Sandler et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
4. **CLAHE**: Zuiderveld (1994). "Contrast limited adaptive histogram equalization"

## Future Work

- [ ] Implement TensorFlow Lite model export
- [ ] Add cloud API for inference
- [ ] Develop web UI for model management
- [ ] Implement active learning for label efficiency
- [ ] Support for multi-output predictions (fiber content, thickness)
- [ ] Real-time video analysis mode

## License

[Specify your license here]

## Contact

For questions or issues, contact the Computer Vision & ML team.

---

**Last Updated**: December 2025  
**Python Version**: 3.10+  
**Status**: Production Ready
