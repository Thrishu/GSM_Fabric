"""
Dataset loader for FabricNet with GSM labels from Excel file.

Loads fabric images and their corresponding Specific Mass (GSM) values
from the FabricNet_parameters.xlsx file.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FabricNetDataset:
    """
    Load FabricNet dataset with real GSM labels from Excel file.

    Loads fabric images (W001.jpg - W130.jpg) and their corresponding
    Specific Mass (GSM) values from FabricNet_parameters.xlsx.
    """

    # Common fabric microscopy image extensions
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, dataset_root: Path, excel_file: str = "FabricNet_parameters.xlsx", random_seed: int = 42):
        """
        Initialize dataset loader.

        Args:
            dataset_root: Root directory of FabricNet dataset
            excel_file: Name of Excel file with GSM labels
            random_seed: For reproducible train/test splits
        """
        self.dataset_root = Path(dataset_root)
        self.excel_file = self.dataset_root / excel_file
        self.random_seed = random_seed
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_file}")

        # Load labels from Excel
        self.labels_df = self._load_labels()
        
        self.logger.info(f"FabricNetDataset initialized: {len(self.labels_df)} samples")

    def _load_labels(self) -> pd.DataFrame:
        """
        Load GSM labels from Excel file.

        Expected columns:
        - Image id: Image identifier (1-130)
        - Specific Mass: GSM value
        - Warp, Weft: Thread counts
        - Texture: Fabric texture type

        Returns:
            DataFrame with image names and labels
        """
        df = pd.read_excel(self.excel_file)
        
        # Validate required columns
        required_cols = ['Image id', 'Specific Mass']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in Excel file: {missing_cols}")
        
        # Create image filename column (W001.jpg, W002.jpg, etc.)
        df['image_filename'] = df['Image id'].apply(lambda x: f"W{x:03d}.jpg")
        
        # Validate that images exist
        valid_rows = []
        for idx, row in df.iterrows():
            image_path = self.dataset_root / row['image_filename']
            if image_path.exists():
                valid_rows.append(idx)
            else:
                self.logger.warning(f"Image not found: {row['image_filename']}")
        
        df = df.loc[valid_rows].reset_index(drop=True)
        
        self.logger.info(f"Loaded {len(df)} samples with GSM labels")
        self.logger.info(f"GSM range: {df['Specific Mass'].min():.0f} - {df['Specific Mass'].max():.0f} g/m²")
        
        if 'Texture' in df.columns:
            texture_counts = df['Texture'].value_counts()
            self.logger.info(f"Texture types: {texture_counts.to_dict()}")
        
        return df

    def get_all_samples(self) -> List[Tuple[Path, float, Dict]]:
        """
        Get all image samples with labels.

        Returns:
            List of (image_path, gsm_value, metadata) tuples
            metadata contains: warp, weft, texture, image_id
        """
        samples = []
        
        for _, row in self.labels_df.iterrows():
            image_path = self.dataset_root / row['image_filename']
            gsm_value = float(row['Specific Mass'])
            
            metadata = {
                'image_id': int(row['Image id']),
                'warp': int(row['Warp']) if 'Warp' in row else None,
                'weft': int(row['Weft']) if 'Weft' in row else None,
                'texture': row['Texture'] if 'Texture' in row else None,
            }
            
            samples.append((image_path, gsm_value, metadata))
        
        return samples

    def split_samples(
        self, samples: List[Tuple[Path, float, Dict]], train_ratio: float = 0.7,
        val_ratio: float = 0.15, test_ratio: float = 0.15, random_seed: int = 42
    ) -> Tuple[list, list, list]:
        """
        Split samples into train/validation/test sets.

        Args:
            samples: List of (image_path, gsm_value, metadata) tuples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            random_seed: For reproducibility

        Returns:
            Tuple of (train_samples, val_samples, test_samples)

        Raises:
            ValueError: If ratios don't sum to ~1.0
        """
        total_ratio = train_ratio + val_ratio + test_ratio

        if not np.isclose(total_ratio, 1.0, atol=0.01):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio}"
            )

        rng = np.random.RandomState(random_seed)

        # Shuffle samples
        samples_copy = samples.copy()
        rng.shuffle(samples_copy)

        # Calculate split sizes
        n_total = len(samples_copy)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split
        train_samples = samples_copy[:n_train]
        val_samples = samples_copy[n_train : n_train + n_val]
        test_samples = samples_copy[n_train + n_val :]

        self.logger.info(
            f"Split: {len(train_samples)} train, {len(val_samples)} val, "
            f"{len(test_samples)} test"
        )

        return train_samples, val_samples, test_samples

    def get_dataset_info(self) -> Dict:
        """
        Get summary information about dataset.

        Returns:
            Dictionary with dataset statistics
        """
        all_samples = self.get_all_samples()

        gsm_values = [gsm for _, gsm, _ in all_samples]
        textures = [meta['texture'] for _, _, meta in all_samples if meta['texture']]

        info = {
            "total_samples": len(all_samples),
            "gsm_min": float(np.min(gsm_values)),
            "gsm_max": float(np.max(gsm_values)),
            "gsm_mean": float(np.mean(gsm_values)),
            "gsm_std": float(np.std(gsm_values)),
            "texture_types": list(set(textures)) if textures else [],
        }

        return info
