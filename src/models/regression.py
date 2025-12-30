"""
Regression models for GSM prediction.

Implements Random Forest and Gradient Boosting regressors.
Provides unified interface for model training, evaluation, and inference.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class GSMRegressor:
    """
    Unified interface for GSM regression models.

    Supports Random Forest and Gradient Boosting regressors.
    Handles training, evaluation, and inference with metrics computation.
    """

    def __init__(
        self,
        model_type: Literal["random_forest", "gradient_boosting"] = "random_forest",
        **kwargs,
    ):
        """
        Initialize regressor.

        Args:
            model_type: "random_forest" or "gradient_boosting"
            **kwargs: Model-specific hyperparameters
        """
        self.model_type = model_type
        self.logger = logging.getLogger(self.__class__.__name__)

        if model_type == "random_forest":
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self._is_fitted = False
        self.logger.info(f"Initialized {model_type} regressor")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train regression model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)

        Raises:
            ValueError: If data shapes are incompatible
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Mismatched sample counts: {X_train.shape[0]} vs {y_train.shape[0]}"
            )

        self.model.fit(X_train, y_train)
        self._is_fitted = True

        self.logger.info(
            f"Model trained on {X_train.shape[0]} samples with {X_train.shape[1]} features"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict GSM values.

        Args:
            X: Input features (n_samples, n_features) or (n_features,)

        Returns:
            Predicted GSM values

        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Handle single sample
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
            predictions = self.model.predict(X)
            return predictions[0]

        return self.model.predict(X)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, metrics: list = ["mae", "rmse", "r2"]
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Computes Mean Absolute Error, Root Mean Squared Error, and R² score.
        Higher R² is better (perfect: 1.0); lower MAE/RMSE is better.

        Args:
            X_test: Test features (n_samples, n_features)
            y_test: True test targets (n_samples,)
            metrics: Which metrics to compute

        Returns:
            Dictionary mapping metric names to values

        Raises:
            RuntimeError: If model not fitted
            ValueError: If unknown metric requested
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X_test)
        results = {}

        for metric in metrics:
            if metric.lower() == "mae":
                results["mae"] = float(mean_absolute_error(y_test, y_pred))
            elif metric.lower() == "rmse":
                results["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            elif metric.lower() == "r2":
                results["r2"] = float(r2_score(y_test, y_pred))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        self.logger.info(f"Evaluation metrics: {results}")

        return results

    def get_feature_importance(self, top_n: int = 10) -> Dict[int, float]:
        """
        Get feature importance scores.

        Supported for Random Forest and Gradient Boosting.
        Rankings based on feature's contribution to split decisions.

        Args:
            top_n: Return top N most important features

        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importances = self.model.feature_importances_

        # Get top N indices
        top_indices = np.argsort(importances)[-top_n:][::-1]

        importance_dict = {
            int(idx): float(importances[idx]) for idx in top_indices
        }

        self.logger.info(f"Top {top_n} important features: {importance_dict}")

        return importance_dict

    def save_model(self, save_path: Path) -> None:
        """
        Serialize trained model to disk.

        Uses pickle format (sklearn standard).

        Args:
            save_path: Path to save model

        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, model_path: Path) -> None:
        """
        Load trained model from disk.

        Args:
            model_path: Path to model pickle file

        Raises:
            FileNotFoundError: If model file not found
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self._is_fitted = True
        self.logger.info(f"Model loaded from {model_path}")
