"""
Feature fusion and scaling module.

Combines texture and CNN features into unified representation.
Applies standardization to normalize feature magnitudes.
"""

import logging
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureFusion:
    """
    Fuse texture and deep CNN features with proper scaling.

    Texture features (GLCM, LBP) and deep CNN features often have
    different magnitude ranges. Scaling ensures fair contribution
    to downstream regression model.
    """

    def __init__(
        self,
        scaler_type: Literal["standard", "minmax"] = "standard",
        save_scaler: bool = True,
        scaler_path: Path = None,
    ):
        """
        Initialize feature fusion.

        Args:
            scaler_type: "standard" (z-score) or "minmax" ([0, 1])
            save_scaler: Whether to persist scaler for inference
            scaler_path: Path to save/load scaler object
        """
        self.scaler_type = scaler_type
        self.save_scaler = save_scaler
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.logger = logging.getLogger(self.__class__.__name__)

        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self._is_fitted = False

    def fuse_features(
        self, texture_features: np.ndarray, deep_features: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate texture and deep features.

        Simple concatenation: preserves all feature information.
        Scaling applied after fusion.

        Args:
            texture_features: Texture feature vector (n_texture_features,)
            deep_features: Deep CNN feature vector (n_deep_features,)

        Returns:
            Fused feature vector
        """
        if texture_features.ndim != 1 or deep_features.ndim != 1:
            raise ValueError("Features must be 1D vectors")

        # Concatenate along feature dimension
        fused = np.concatenate([texture_features, deep_features])

        return fused

    def fit(self, features: np.ndarray) -> None:
        """
        Fit scaler on training features.

        Args:
            features: Training feature matrix (n_samples, n_features)
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {features.shape}")

        self.scaler.fit(features)
        self._is_fitted = True

        self.logger.info(f"Scaler fitted on {features.shape[0]} samples with {features.shape[1]} features")

        if self.save_scaler and self.scaler_path:
            self._save_scaler()

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Scale features using fitted scaler.

        Args:
            features: Feature matrix to scale (n_samples, n_features) or (n_features,)

        Returns:
            Scaled features with same shape

        Raises:
            RuntimeError: If scaler not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        # Handle single sample
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)
            scaled = self.scaler.transform(features)
            return scaled[0]

        scaled = self.scaler.transform(features)
        return scaled

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform in one step.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Scaled features
        """
        self.fit(features)
        return self.transform(features)

    def _save_scaler(self) -> None:
        """
        Serialize scaler to disk for inference.

        Uses pickle format (sklearn standard).
        """
        import pickle

        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        self.logger.info(f"Scaler saved to {self.scaler_path}")

    def load_scaler(self, scaler_path: Path) -> None:
        """
        Load scaler from disk.

        Args:
            scaler_path: Path to scaler pickle file

        Raises:
            FileNotFoundError: If scaler file not found
        """
        import pickle

        scaler_path = Path(scaler_path)

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self._is_fitted = True
        self.logger.info(f"Scaler loaded from {scaler_path}")

    def get_feature_info(
        self, texture_n_features: int, deep_n_features: int
    ) -> Tuple[int, int, int]:
        """
        Get information about fused features.

        Args:
            texture_n_features: Number of texture features
            deep_n_features: Number of deep CNN features

        Returns:
            Tuple of (total_features, texture_start_idx, deep_start_idx)
        """
        total = texture_n_features + deep_n_features
        return total, 0, texture_n_features
