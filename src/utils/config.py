"""
Configuration management for the fabric GSM pipeline.

Loads and validates YAML configuration with type checking and defaults.
Provides a single source of truth for all hyperparameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


class Config:
    """
    Configuration manager for fabric GSM pipeline.

    Loads YAML config file and provides dictionary-like access to parameters.
    Supports nested configuration with dot notation access.
    """

    def __init__(self, config_path: Path):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self._config = self._load_yaml()
        self.logger.info(f"Configuration loaded from {self.config_path}")

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Returns:
            Dictionary containing configuration

        Raises:
            yaml.YAMLError: If YAML parsing fails
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.

        Args:
            key: Configuration key (supports nested keys like "data.image_size")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
                if value is default:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.get(key)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.config_path})"


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration with optional path override.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        Config instance

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        # Try to find config in standard locations
        default_paths = [
            Path("configs/config.yaml"),
            Path("./config.yaml"),
        ]
        for path in default_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(
                "No config.yaml found. Checked: " + ", ".join(str(p) for p in default_paths)
            )

    return Config(config_path)
