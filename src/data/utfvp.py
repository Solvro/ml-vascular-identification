"""
UTFVP (University of Twente Finger Vascular Pattern) dataset implementation.

Multi-session finger vein dataset with detailed session and timestamp tracking.
"""
import re
from pathlib import Path

import pandas as pd

from .base import BaseDataset, BaseScanner, build_manifest_cache

# Default data paths - resolve relative to project root
_PROJECT_ROOT = Path(
    __file__
).parent.parent.parent  # Go up from src/data/ to project root
DEFAULT_UTFVP_PATH = str(_PROJECT_ROOT / "data" / "UTFVP")
DEFAULT_CACHE_DIR = str(_PROJECT_ROOT / "data" / "cache")


class UTFVPScanner(BaseScanner):
    """Scanner for UTFVP dataset layout."""

    @staticmethod
    def scan(root: str) -> pd.DataFrame:
        """Scan the UTFVP dataset layout.

        Expects dataset/data folder with per-patient subfolders containing images.
        Naming: {patient_id}_{finger}_{session}_{timestamp}.png
        Example: 0001_1_1_120509-135315.png
          - patient: 0001
          - finger: 1-6
          - session: 1-2
          - timestamp: YYMMDD-HHMMSS

        Args:
            root: Root directory of the UTFVP dataset.

        Returns:
            DataFrame with columns: ['path', 'patient_id', 'finger', 'session',
                                    'timestamp', 'finger_class_id', 'dataset'].
        """
        rows = []
        root = Path(root)
        data_dir = root / "dataset" / "data"

        if not data_dir.exists():
            data_dir = root / "data"
        if not data_dir.exists():
            data_dir = root

        # Pattern: {patient}_{finger}_{session}_{timestamp}.png
        # Example: 0001_1_1_120509-135315.png
        pattern = re.compile(r"(\d{4})_(\d)_(\d)_(\d+-\d+)\.png", re.IGNORECASE)

        for patient_dir in sorted(data_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name.zfill(4)

            for img_file in sorted(patient_dir.iterdir()):
                if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
                    continue

                match = pattern.match(img_file.name)
                if not match:
                    continue

                pid, finger, session, timestamp = match.groups()
                patient_id = pid.zfill(4)
                finger = int(finger)
                session = int(session)

                # Create unique finger class ID: dataset_patient_finger
                finger_class_id = f"utfvp_{patient_id}_f{finger}"

                rows.append(
                    {
                        "path": str(img_file),
                        "patient_id": patient_id,
                        "finger": finger,
                        "session": session,
                        "timestamp": timestamp,
                        "finger_class_id": finger_class_id,
                        "dataset": "utfvp",
                    }
                )

        return pd.DataFrame(rows)


class UTFVPDataset(BaseDataset):
    """Dataset class for UTFVP multi-session finger vein images."""

    def __init__(self, df=None, transform=None, label_encoder=None, cfg=None):
        """Initialize UTFVP dataset.

        Args:
            df: Optional DataFrame. If None, will auto-load from DEFAULT_UTFVP_PATH.
            transform: Optional image transform.
            label_encoder: Optional label encoder.
            cfg: Optional config object. If provided, will auto-build transform.
        """
        if df is None:
            df = build_utfvp_manifest()

        # Auto-build transform from config if provided
        if cfg is not None and transform is None:
            from .transforms import build_transforms_from_config

            transform = build_transforms_from_config(cfg)

        super().__init__(df, transform, label_encoder)

    def _get_metadata(self, row):
        """Extract UTFVP-specific metadata from DataFrame row.

        Args:
            row: Pandas Series representing one sample.

        Returns:
            dict: Metadata with finger, session and timestamp information.
        """
        metadata = {
            "finger_class_id": row["finger_class_id"],
            "patient_id": row["patient_id"],
            "finger": row["finger"],
            "session": row["session"],
            "timestamp": row["timestamp"],
            "dataset": row.get("dataset", "utfvp"),
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
        """Get unique finger IDs in the dataset."""
        if "finger" in self.df.columns:
            return sorted(self.df["finger"].unique().tolist())
        return []

    def get_sessions(self):
        """Get unique sessions in the dataset."""
        if "session" in self.df.columns:
            return sorted(self.df["session"].unique().tolist())
        return []

    def get_samples_by_finger(self, finger: int):
        """Get all samples for a specific finger."""
        if "finger" not in self.df.columns:
            raise ValueError("Dataset does not have 'finger' column")
        return self.df[self.df["finger"] == finger]

    def get_samples_by_session(self, session: int):
        """Get all samples for a specific session."""
        if "session" not in self.df.columns:
            raise ValueError("Dataset does not have 'session' column")
        return self.df[self.df["session"] == session]


def build_utfvp_manifest(cache_dir: str = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Build (or load) a cached manifest for UTFVP dataset from default paths.

    Args:
        cache_dir: Directory for cached parquet manifests.

    Returns:
        DataFrame with per-image rows and UTFVP-specific columns.
    """
    return build_manifest_cache("utfvp", DEFAULT_UTFVP_PATH, UTFVPScanner, cache_dir)
