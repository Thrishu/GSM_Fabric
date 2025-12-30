"""
Deep feature extraction using pretrained CNN.

MobileNetV2/V3 models provide efficient feature extraction suitable for
resource-constrained environments (e.g., Raspberry Pi).
"""

import logging
from typing import Literal, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (  # type: ignore
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Small,
)
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore


class DeepFeatureExtractor:
    """
    Extract features from pretrained CNN backbone.

    Uses MobileNetV2/V3 for lightweight feature extraction.
    Removes classification head and uses global average pooling.
    Features are 1D vectors suitable for downstream regression.
    """

    def __init__(
        self,
        model_type: Literal["MobileNetV2", "MobileNetV3Small", "MobileNetV3Large"] = "MobileNetV2",
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        weights: str = "imagenet",
        pooling: str = "global_average",
        preprocessing_mode: str = "tf",
        use_gpu: bool = True,
    ):
        """
        Initialize deep feature extractor.

        Args:
            model_type: CNN architecture ("MobileNetV2", "MobileNetV3Small", "MobileNetV3Large")
            input_shape: Input tensor shape (height, width, channels)
            weights: Pretrained weights ("imagenet" or None)
            pooling: Pooling method ("global_average" or "global_max")
            preprocessing_mode: Input normalization ("tf", "torch", "torch_agnostic")
            use_gpu: Whether to use GPU acceleration
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.weights = weights
        self.pooling = pooling
        self.preprocessing_mode = preprocessing_mode
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configure device placement
        if use_gpu:
            self.device = "/gpu:0"
        else:
            self.device = "/cpu:0"

        self.model = self._build_model()
        self.logger.info(f"DeepFeatureExtractor initialized with {model_type}")

    def _build_model(self) -> Model:
        """
        Build feature extraction model.

        Removes classification head and applies global average pooling
        to convert spatial features to 1D vector.

        Returns:
            Keras model that outputs 1D feature vectors

        Raises:
            ValueError: If model_type is unknown
        """
        # Select base model
        if self.model_type == "MobileNetV2":
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                weights=self.weights,
                include_top=False,
            )
        elif self.model_type == "MobileNetV3Small":
            base_model = MobileNetV3Small(
                input_shape=self.input_shape,
                weights=self.weights,
                include_top=False,
            )
        elif self.model_type == "MobileNetV3Large":
            base_model = MobileNetV3Large(
                input_shape=self.input_shape,
                weights=self.weights,
                include_top=False,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Freeze pretrained weights (transfer learning)
        base_model.trainable = False

        # Add global average pooling
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        outputs = x

        model = Model(inputs=inputs, outputs=outputs)

        self.logger.info(f"Feature vector dimension: {model.output_shape[-1]}")

        return model

    def _preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Applies model-specific normalization:
        - "tf": (x / 127.5) - 1 (ImageNet, range [-1, 1])
        - "torch": (x - mean) / std with PyTorch ImageNet stats

        Args:
            image: Input image (float32, range [0, 1])

        Returns:
            Preprocessed image for model input
        """
        if self.preprocessing_mode == "tf":
            # TensorFlow preprocessing: scale to [-1, 1]
            processed = (image / 0.5) - 1.0

        elif self.preprocessing_mode == "torch":
            # PyTorch ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            processed = (image - mean) / std

        elif self.preprocessing_mode == "torch_agnostic":
            # Conservative: just rescale to [-1, 1]
            processed = (image / 0.5) - 1.0

        else:
            self.logger.warning(f"Unknown preprocessing mode: {self.preprocessing_mode}")
            processed = image

        return processed.astype(np.float32)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from single image.

        Image must already be:
        - Preprocessed (normalized to [0, 1])
        - Resized to 224x224
        - Converted to grayscale and stacked to 3 channels (for RGB input)

        Args:
            image: Input image (float32, shape (224, 224) or (224, 224, 1))

        Returns:
            1D feature vector (float32)

        Raises:
            ValueError: If image shape is incompatible
        """
        # Handle grayscale: convert to 3-channel by replication
        if len(image.shape) == 2:
            # Expand and replicate channels
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:
            # Replicate single channel
            image = np.concatenate([image, image, image], axis=-1)
        elif image.shape[-1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {image.shape[-1]}")

        # Preprocess
        image = self._preprocess_input(image)

        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)

        # Extract features with device placement
        with tf.device(self.device):
            features = self.model(image_batch, training=False)

        # Convert to numpy and squeeze batch dimension
        features = features.numpy()[0]

        return features.astype(np.float32)

    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from batch of images.

        More efficient than calling extract_features() on each image.

        Args:
            images: Batch of images (shape: (batch_size, 224, 224) or (batch_size, 224, 224, 1))

        Returns:
            Feature matrix (shape: (batch_size, feature_dim))
        """
        batch_size = images.shape[0]
        features_list = []

        for i in range(batch_size):
            features = self.extract_features(images[i])
            features_list.append(features)

        features_batch = np.array(features_list, dtype=np.float32)
        return features_batch
