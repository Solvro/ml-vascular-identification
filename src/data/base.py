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
    """Abstract base class for image datasets with common functionality.
    
    Supports both closed-set (patient-level) and open-set (finger-level) recognition.
    """

    def __init__(
        self,
        df,
        transform=None,
        label_encoder=None,
        use_finger_classes=False,
    ):
        """Initialize base dataset.

        Args:
            df: Pandas DataFrame with required columns:
                - For closed-set: ["path", "patient_id"]
                - For open-set: ["path", "patient_id", "finger_class_id"]
            transform: Optional callable applied to PIL image.
            label_encoder: Optional dict {key: int}; built if None.
            use_finger_classes: If True, use finger_class_id for labels (OpenSet mode).
                               If False, use patient_id for labels (closed-set mode).
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_finger_classes = use_finger_classes

        # Determine which identifier to use for labels
        if use_finger_classes:
            if "finger_class_id" not in df.columns:
                raise ValueError(
                    "use_finger_classes=True requires 'finger_class_id' column"
                )
            id_column = "finger_class_id"
        else:
            id_column = "patient_id"

        # Build label encoder
        if label_encoder is None:
            ids = sorted(self.df[id_column].unique().tolist())
            self.le = {id_val: i for i, id_val in enumerate(ids)}  # contiguous labels
        else:
            self.le = label_encoder

        self.id_column = id_column

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

        # Get label based on mode (finger_class_id for OpenSet, patient_id for closed-set)
        label_key = row[self.id_column]
        label = self.le[label_key]
        
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
    
    def get_finger_classes(self):
        """Get unique finger class IDs in the dataset (for OpenSet mode).
        
        Returns:
            List of finger_class_id strings, or empty list if column doesn't exist.
        """
        if "finger_class_id" in self.df.columns:
            return self.df["finger_class_id"].unique().tolist()
        return []
    
    def get_samples_by_finger_class(self, finger_class_id):
        """Get all samples for a specific finger class.
        
        Args:
            finger_class_id: Finger class identifier (e.g., 'mmcbnu_001_L_Fore').
            
        Returns:
            DataFrame subset with all samples for this finger class.
        """
        if "finger_class_id" not in self.df.columns:
            raise ValueError("Dataset does not have 'finger_class_id' column")
        return self.df[self.df["finger_class_id"] == finger_class_id]
    
    def filter_by_openset_split(self, split: str):
        """Create a new dataset filtered by openset_split ('known' or 'unknown').
        
        Args:
            split: Either 'known' or 'unknown'.
            
        Returns:
            New dataset instance with filtered DataFrame.
        """
        if "openset_split" not in self.df.columns:
            raise ValueError("Dataset does not have 'openset_split' column")
        
        if split not in ["known", "unknown"]:
            raise ValueError(f"split must be 'known' or 'unknown', got '{split}'")
        
        filtered_df = self.df[self.df["openset_split"] == split].copy().reset_index(drop=True)
        
        # Create new instance with same type and settings
        return self.__class__(
            df=filtered_df,
            transform=self.transform,
            label_encoder=self.le,
            use_finger_classes=self.use_finger_classes,
        )
    
    def get_num_classes(self):
        """Get number of unique classes in the dataset.
        
        Returns:
            Number of unique classes (finger classes or patients depending on mode).
        """
        return len(self.le)


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
