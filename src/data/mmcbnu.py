"""
MMCBNU dataset implementation.

Handles the MMCBNU dataset with per-patient folders containing per-finger subfolders with images.
"""
from pathlib import Path

import pandas as pd

from .base import BaseDataset, BaseScanner, build_manifest_cache

# Default data paths - resolve relative to project root
_PROJECT_ROOT = Path(
    __file__
).parent.parent.parent  # Go up from src/data/ to project root
DEFAULT_MMCBNU_PATH = str(_PROJECT_ROOT / "data" / "mmcbnu")
DEFAULT_CACHE_DIR = str(_PROJECT_ROOT / "data" / "cache")


class MMCBNUScanner(BaseScanner):
    """Scanner for MMCBNU dataset layout."""

    @staticmethod
    def scan(root: str) -> pd.DataFrame:
        """Scan the 'mmcbnu' dataset layout.

        Expects ROIs folder with per-patient folders, each containing per-finger subfolders with images.

        Args:
            root: Root directory of the MMCBNU dataset.

        Returns:
            DataFrame with columns: ['path', 'patient_id', 'finger', 'finger_class_id', 'dataset'].
        """
        rows = []
        root = Path(root)

        # Look for ROIs subdirectory
        rois_dir = root / "ROIs"
        if not rois_dir.exists():
            # Fallback to direct patient folders
            rois_dir = root

        for pid_dir in rois_dir.iterdir():
            if not pid_dir.is_dir():
                continue
            pid = pid_dir.name.zfill(3)

            for finger_dir in pid_dir.iterdir():
                if not finger_dir.is_dir():
                    continue
                finger_name = finger_dir.name
                # Create unique finger class ID: dataset_patientID_finger
                finger_class_id = f"mmcbnu_{pid}_{finger_name}"
                
                for f in finger_dir.iterdir():
                    if f.suffix.lower() not in [".bmp", ".png", ".jpg", ".jpeg"]:
                        continue
                    rows.append(
                        {
                            "path": str(f),
                            "patient_id": pid,
                            "finger": finger_name,
                            "finger_class_id": finger_class_id,
                            "dataset": "mmcbnu",
                        }
                    )

        return pd.DataFrame(rows)


class MMCBNUDataset(BaseDataset):
    """Dataset class for MMCBNU vascular images."""

    def __init__(self, df=None, transform=None, label_encoder=None, cfg=None):
        """Initialize MMCBNU dataset.

        Args:
            df: Optional DataFrame. If None, will auto-load from DEFAULT_MMCBNU_PATH.
            transform: Optional image transform.
            label_encoder: Optional label encoder.
            cfg: Optional config object. If provided, will auto-build transform.
        """
        if df is None:
            df = build_mmcbnu_manifest()

        # Auto-build transform from config if provided
        if cfg is not None and transform is None:
            from .transforms import build_transforms_from_config

            transform = build_transforms_from_config(cfg)

        super().__init__(df, transform, label_encoder)

    def _get_metadata(self, row):
        """Extract MMCBNU-specific metadata from DataFrame row.

        Args:
            row: Pandas Series representing one sample.

        Returns:
            dict: Metadata with finger and OpenSet information.
        """
        metadata = {
            "finger_class_id": row["finger_class_id"],
            "patient_id": row["patient_id"],
            "finger": row["finger"],
            "dataset": row["dataset"],
            "path": row["path"],
        }
        
        # Add openset_split if available
        if "openset_split" in row:
            metadata["openset_split"] = row["openset_split"]
        
        # Add sample_split if available (for enrollment/test)
        if "sample_split" in row:
            metadata["sample_split"] = row["sample_split"]
            
        return metadata

    def get_fingers(self):
        """Get unique finger types in the dataset."""
        return self.df["finger"].unique().tolist()

    def get_samples_by_finger(self, finger):
        """Get all samples for a specific finger type."""
        return self.df[self.df["finger"] == finger]


def build_mmcbnu_manifest() -> pd.DataFrame:
    """Build (or load) a cached manifest for MMCBNU dataset from default paths.

    Returns:
        DataFrame with per-image rows and MMCBNU-specific columns.
    """
    return build_manifest_cache(
        "mmcbnu", DEFAULT_MMCBNU_PATH, MMCBNUScanner, DEFAULT_CACHE_DIR
    )
