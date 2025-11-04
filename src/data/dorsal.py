"""
Dorsal dataset implementation.

Handles the dorsal vascular dataset with patient folders containing Left/Right images.
"""
import re
from pathlib import Path

import pandas as pd

from .base import BaseDataset, BaseScanner, build_manifest_cache

# Default data paths - resolve relative to project root
_PROJECT_ROOT = Path(
    __file__
).parent.parent.parent  # Go up from src/data/ to project root
DEFAULT_DORSAL_PATH = str(_PROJECT_ROOT / "data" / "dorsal")
DEFAULT_CACHE_DIR = str(_PROJECT_ROOT / "data" / "cache")


class DorsalScanner(BaseScanner):
    """Scanner for dorsal dataset layout."""

    @staticmethod
    def scan(root: str) -> pd.DataFrame:
        """Scan the 'dorsal' dataset layout.

        Expects P### folders with 'Left'/'Right' subfolders containing images.

        Args:
            root: Root directory of the dorsal dataset.

        Returns:
            DataFrame with columns: ['path', 'patient_id', 'side', 'dataset'].
        """
        rows = []
        root = Path(root)
        p_pat = re.compile(
            r"P(\d+)", re.IGNORECASE
        )  # match patient folder names like 'P001'

        for p_dir in root.iterdir():
            if not p_dir.is_dir():
                continue  # skip files
            m = p_pat.match(p_dir.name)
            if not m:
                continue  # skip non-patient dirs
            pid = m.group(1).zfill(3)

            for side in ["Left", "Right"]:
                sdir = p_dir / side
                if not sdir.exists():
                    continue
                # Create unique hand-level class id: dorsal_<patient>_<side>
                finger_class_id = f"dorsal_{pid}_{side}"

                for f in sdir.iterdir():
                    if f.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
                        continue
                    rows.append(
                        {
                            "path": str(f),
                            "patient_id": pid,
                            "side": side,
                            "finger_class_id": finger_class_id,
                            "dataset": "dorsal",
                        }
                    )

        return pd.DataFrame(rows)


class DorsalDataset(BaseDataset):
    """Dataset class for dorsal vascular images."""

    def __init__(self, df=None, transform=None, label_encoder=None, cfg=None):
        """Initialize dorsal dataset.

        Args:
            df: Optional DataFrame. If None, will auto-load from DEFAULT_DORSAL_PATH.
            transform: Optional image transform.
            label_encoder: Optional label encoder.
            cfg: Optional config object. If provided, will auto-build transform.
        """
        if df is None:
            df = build_dorsal_manifest()

        # Auto-build transform from config if provided
        if cfg is not None and transform is None:
            from .transforms import build_transforms_from_config

            transform = build_transforms_from_config(cfg)

        super().__init__(df, transform, label_encoder)

    def _get_metadata(self, row):
        """Extract dorsal-specific metadata from DataFrame row.

        Args:
            row: Pandas Series representing one sample.

        Returns:
            dict: Metadata with side information.
        """
        metadata = {
            "finger_class_id": row.get("finger_class_id"),
            "patient_id": row["patient_id"],
            "side": row["side"],
            "dataset": row.get("dataset", "dorsal"),
            "path": row["path"],
        }

        # Include openset/session split markers if present in manifest
        if "openset_split" in row:
            metadata["openset_split"] = row["openset_split"]
        if "sample_split" in row:
            metadata["sample_split"] = row["sample_split"]

        return metadata


def build_dorsal_manifest() -> pd.DataFrame:
    """Build (or load) a cached manifest for dorsal dataset from default paths.

    Returns:
        DataFrame with per-image rows and dorsal-specific columns.
    """
    return build_manifest_cache(
        "dorsal", DEFAULT_DORSAL_PATH, DorsalScanner, DEFAULT_CACHE_DIR
    )
