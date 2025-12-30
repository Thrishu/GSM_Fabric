"""
Image preprocessing module for fabric microscopy images.

Handles grayscale conversion, resizing, histogram equalization, and normalization.
All operations preserve image quality while standardizing input dimensions.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """
    Preprocessing pipeline for fabric microscopy images.

    Applies a sequence of transformations:
    1. Load image (BGR to RGB/Grayscale)
    2. Resize to target dimensions
    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Normalize pixel values to [0, 1]
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        normalization_mode: str = "minmax",
    ):
        """
        Initialize preprocessor with specified parameters.

        Args:
            target_size: Target image dimensions (height, width)
            enable_clahe: Whether to apply CLAHE enhancement
            clahe_clip_limit: CLAHE contrast limit (2.0-4.0 typical)
            clahe_tile_size: CLAHE grid size for local processing
            normalization_mode: "minmax" (to [0,1]) or "standardization" (z-score)
        """
        self.target_size = target_size
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.normalization_mode = normalization_mode
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize CLAHE object if enabled
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_size
            )
        else:
            self.clahe = None

    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (BGR format from OpenCV)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.

        Assumes input is BGR format (OpenCV standard).
        Uses standard luminosity weights for proper conversion.

        Args:
            image: Input image (BGR)

        Returns:
            Grayscale image (uint8)
        """
        # Check if already grayscale
        if len(image.shape) == 2:
            return image

        # Convert BGR to grayscale using luminosity formula
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions.

        Uses cubic interpolation for high-quality resizing.
        Aspect ratio may not be preserved; images are directly resized.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
        return resized

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.

        CLAHE enhances local contrast while preventing noise amplification.
        Critical for microscopy images with varying illumination.

        Args:
            image: Grayscale image (uint8)

        Returns:
            CLAHE-enhanced image (uint8)
        """
        if self.clahe is None:
            return image

        enhanced = self.clahe.apply(image)
        return enhanced

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values.

        Args:
            image: Input image (uint8, values 0-255)

        Returns:
            Normalized image (float32, values in specified range)
        """
        image = image.astype(np.float32)

        if self.normalization_mode == "minmax":
            # Scale to [0, 1]
            image = image / 255.0

        elif self.normalization_mode == "standardization":
            # Z-score normalization (mean=0, std=1)
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
            else:
                self.logger.warning("Standard deviation is zero during normalization")

        else:
            raise ValueError(f"Unknown normalization mode: {self.normalization_mode}")

        return image

    def preprocess(self, image_path: Path) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Pipeline:
        1. Load image
        2. Convert to grayscale
        3. Resize to target dimensions
        4. Apply CLAHE enhancement
        5. Normalize

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image (float32, normalized)

        Raises:
            FileNotFoundError: If image file not found
            ValueError: If image cannot be loaded or processed
        """
        # Load
        image = self.load_image(image_path)

        # Grayscale
        image = self.to_grayscale(image)

        # Resize
        image = self.resize(image)

        # CLAHE
        image = self.apply_clahe(image)

        # Normalize
        image = self.normalize(image)

        return image

    def preprocess_array(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to already-loaded image array.

        Useful when image is already in memory (e.g., from dataset).

        Args:
            image: Input image array (BGR or grayscale)

        Returns:
            Preprocessed image (float32, normalized)
        """
        # Grayscale
        image = self.to_grayscale(image)

        # Resize
        image = self.resize(image)

        # CLAHE
        image = self.apply_clahe(image)

        # Normalize
        image = self.normalize(image)

        return image
