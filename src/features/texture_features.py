"""
Texture feature extraction module using GLCM and Local Binary Pattern.

Extracts low-level statistical texture descriptors from fabric images.
Features are designed to capture fabric structure, weave patterns, and surface properties.
"""

import logging
from typing import List, Tuple

import numpy as np
from skimage import feature
from skimage.feature import graycomatrix, graycoprops


class TextureFeatureExtractor:
    """
    Extract texture features using GLCM and LBP methods.

    GLCM (Gray Level Co-occurrence Matrix):
    - Measures spatial relationships between pixel intensities
    - Captures contrast, homogeneity, energy, and correlation

    LBP (Local Binary Pattern):
    - Measures local texture patterns with rotation invariance
    - Creates histogram of uniform micro-patterns
    """

    def __init__(
        self,
        glcm_enabled: bool = True,
        glcm_distances: List[int] = [1, 3],
        glcm_angles: List[int] = [0, 45, 90, 135],
        glcm_levels: int = 256,
        glcm_metrics: List[str] = ["contrast", "homogeneity", "energy", "correlation"],
        lbp_enabled: bool = True,
        lbp_radius: int = 3,
        lbp_n_points: int = 8,
        lbp_method: str = "uniform",
        lbp_n_bins: int = 59,
    ):
        """
        Initialize texture feature extractor.

        Args:
            glcm_enabled: Enable GLCM feature extraction
            glcm_distances: Pixel distances for co-occurrence computation
            glcm_angles: Directions in degrees (0, 45, 90, 135)
            glcm_levels: Number of gray levels for quantization
            glcm_metrics: Which GLCM metrics to extract
            lbp_enabled: Enable LBP feature extraction
            lbp_radius: Neighborhood radius for LBP
            lbp_n_points: Number of neighborhood points (typically 8*radius)
            lbp_method: "uniform", "nri_uniform", or "var"
            lbp_n_bins: Number of bins for LBP histogram
        """
        self.glcm_enabled = glcm_enabled
        self.glcm_distances = glcm_distances
        self.glcm_angles = [np.radians(angle) for angle in glcm_angles]
        self.glcm_levels = glcm_levels
        self.glcm_metrics = glcm_metrics
        self.lbp_enabled = lbp_enabled
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.lbp_method = lbp_method
        self.lbp_n_bins = lbp_n_bins
        self.logger = logging.getLogger(self.__class__.__name__)

    def _quantize_image(self, image: np.ndarray, levels: int) -> np.ndarray:
        """
        Quantize image to specified number of levels.

        Reduces computational cost and noise sensitivity for texture analysis.
        Maps pixel values [0, 255] to discrete levels.

        Args:
            image: Input grayscale image (float32, [0, 1])
            levels: Target number of quantization levels

        Returns:
            Quantized image with values in range [0, levels-1]
        """
        # Scale to [0, levels-1]
        quantized = (image * (levels - 1)).astype(np.uint8)
        return quantized

    def extract_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract GLCM (Gray Level Co-occurrence Matrix) features.

        Computes spatial co-occurrence statistics for specified distances and angles.
        Properties measure:
        - contrast: Local variation magnitude
        - homogeneity: Closeness of distribution to diagonal
        - energy: Distribution uniformity
        - correlation: Linear relationship between pixels

        Args:
            image: Grayscale image (float32, [0, 1])

        Returns:
            Feature vector of all GLCM metrics (float32)

        Raises:
            ValueError: If image is invalid or too small
        """
        if image.size == 0:
            raise ValueError("Image is empty")

        if image.shape[0] < 10 or image.shape[1] < 10:
            self.logger.warning(f"Image too small for GLCM: {image.shape}")
            # Return zero features as fallback
            n_features = len(self.glcm_distances) * len(self.glcm_angles) * len(self.glcm_metrics)
            return np.zeros(n_features, dtype=np.float32)

        # Quantize image
        quantized = self._quantize_image(image, self.glcm_levels)

        # Convert angles to list format required by skimage
        angles_list = list(self.glcm_angles)
        distances_list = list(self.glcm_distances)

        try:
            # Compute GLCM
            glcm = graycomatrix(
                quantized,
                distances=distances_list,
                angles=angles_list,
                levels=self.glcm_levels,
                symmetric=True,
                normed=True,
            )

            # Extract properties
            features = []
            for metric in self.glcm_metrics:
                props = graycoprops(glcm, metric)  # Shape: (n_distances, n_angles)
                # Flatten: [dist0_ang0, dist0_ang1, ..., dist1_ang0, ...]
                features.extend(props.ravel())

            glcm_features = np.array(features, dtype=np.float32)
            return glcm_features

        except Exception as e:
            self.logger.error(f"GLCM extraction failed: {e}")
            n_features = len(distances_list) * len(angles_list) * len(self.glcm_metrics)
            return np.zeros(n_features, dtype=np.float32)

    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP (Local Binary Pattern) features.

        Computes uniform LBP histogram capturing local micro-patterns.
        Uniform patterns are binary strings with ≤2 bit transitions,
        capturing edge-like and corner-like structures.

        Args:
            image: Grayscale image (float32, [0, 1])

        Returns:
            LBP histogram (float32, normalized to sum to 1)

        Raises:
            ValueError: If image is invalid
        """
        if image.size == 0:
            raise ValueError("Image is empty")

        try:
            # Compute LBP
            lbp = feature.local_binary_pattern(
                image,
                P=self.lbp_n_points,
                R=self.lbp_radius,
                method=self.lbp_method,
            )

            # Compute histogram
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=self.lbp_n_bins,
                range=(0, self.lbp_n_bins),
            )

            # Normalize histogram to sum to 1
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist = hist / hist.sum()
            else:
                self.logger.warning("LBP histogram is empty")

            return hist

        except Exception as e:
            self.logger.error(f"LBP extraction failed: {e}")
            return np.zeros(self.lbp_n_bins, dtype=np.float32)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all enabled texture features.

        Concatenates GLCM and LBP features into single vector.

        Args:
            image: Grayscale image (float32, normalized to [0, 1])

        Returns:
            Combined texture feature vector (float32)

        Raises:
            ValueError: If image is invalid
        """
        if image.size == 0:
            raise ValueError("Image is empty")

        features = []

        if self.glcm_enabled:
            glcm_feats = self.extract_glcm_features(image)
            features.append(glcm_feats)
            self.logger.debug(f"GLCM features shape: {glcm_feats.shape}")

        if self.lbp_enabled:
            lbp_feats = self.extract_lbp_features(image)
            features.append(lbp_feats)
            self.logger.debug(f"LBP features shape: {lbp_feats.shape}")

        if not features:
            raise ValueError("No features enabled")

        # Concatenate all features
        combined = np.concatenate(features)
        self.logger.debug(f"Total texture features: {combined.shape[0]}")

        return combined
