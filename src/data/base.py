"""
Base classes and utilities for dataset handling.

This module provides abstract base class for dataset implementations.
"""
import os
from abc import ABC, abstractmethod

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BaseScanner(ABC):
    """Abstract base class for dataset scanners."""

    @staticmethod
    @abstractmethod
    def scan(root: str) -> pd.DataFrame:
        """Scan dataset directory structure.

        Args:
            root: Root directory of the dataset.

        Returns:
            DataFrame with dataset-specific columns including at least:
            ['path', 'patient_id', 'dataset']
        """
        pass


class BaseDataset(Dataset, ABC):
    """Abstract base class for image datasets with common functionality."""

    def __init__(self, df, transform=None, label_encoder=None):
        """Initialize base dataset.

        Args:
            df: Pandas DataFrame with at least ["path", "patient_id"] columns.
            transform: Optional callable applied to PIL image.
            label_encoder: Optional dict {patient_id: int}; built if None.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

        if label_encoder is None:
            pids = sorted(self.df["patient_id"].unique().tolist())
            self.le = {pid: i for i, pid in enumerate(pids)}  # contiguous labels
        else:
            self.le = label_encoder

    def __len__(self):
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load item at index: (image, label, metadata).

        Args:
            idx: Index of the sample to load.

        Returns:
            tuple: (image, label, metadata) where metadata contains dataset-specific info.
        """
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("L")  # force grayscale

        if self.transform:
            img = self.transform(img)

        label = self.le[row["patient_id"]]
        metadata = self._get_metadata(row)

        return img, label, metadata

    @abstractmethod
    def _get_metadata(self, row):
        """Extract metadata from DataFrame row.

        Args:
            row: Pandas Series representing one sample.

        Returns:
            dict: Metadata specific to dataset type.
        """
        pass

    def get_patient_ids(self):
        """Get unique patient IDs in the dataset."""
        return self.df["patient_id"].unique().tolist()

    def get_samples_by_patient(self, patient_id):
        """Get all samples for a specific patient."""
        return self.df[self.df["patient_id"] == patient_id]


def build_manifest_cache(
    name: str, root: str, scanner_class, cache_dir: str = "outputs/manifests"
) -> pd.DataFrame:
    """Generic function to build (or load) cached manifest using provided scanner.

    Args:
        name: Dataset name for cache file.
        root: Root directory of the dataset.
        scanner_class: Scanner class with scan() method.
        cache_dir: Directory for cached parquet manifests.

    Returns:
        DataFrame with per-image rows and dataset-specific columns.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{name}.parquet")

    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df = scanner_class.scan(root)
    df.to_parquet(cache_path, index=False)
    return df


def create_dataset_from_config(cfg):
    """Create dataset from config object.

    Args:
        cfg: Config object with cfg.data containing:
            - dataset: 'dorsal' or 'mmcbnu'
            - transforms: transform configuration
            - split: split configuration (optional)

    Returns:
        Dataset instance with transforms applied.
    """
    # Import here to avoid circular imports
    from .dorsal import DorsalDataset
    from .mmcbnu import MMCBNUDataset
    from .splits import make_patient_split
    from .transforms import build_transforms_from_config

    # Get dataset type
    dataset_name = cfg.data.dataset

    # Build transforms
    transform = build_transforms_from_config(cfg)

    # Create base dataset
    if dataset_name == "dorsal":
        dataset = DorsalDataset(transform=transform)
    elif dataset_name == "mmcbnu":
        dataset = MMCBNUDataset(transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Apply splits if configured
    if hasattr(cfg.data, "split"):
        df_with_splits = make_patient_split(
            dataset.df,
            train=cfg.data.split.train,
            val=cfg.data.split.val,
            test=cfg.data.split.test,
            seed=cfg.data.split.seed,
        )
        dataset.df = df_with_splits

    return dataset


# Export main abstract classes
__all__ = [
    "BaseScanner",
    "BaseDataset",
    "build_manifest_cache",
    "create_dataset_from_config",
]
