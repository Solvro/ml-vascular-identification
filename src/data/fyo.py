"""
FYO (Finger vein Online) dataset implementation.

Supports multi-body-part vein imaging: Dorsal, Palm, and Wrist.
Provides unified interface for Open-Set recognition across body parts.
"""
import re
from pathlib import Path

import pandas as pd

from .base import BaseDataset, BaseScanner

# Default data paths - resolve relative to project root
_PROJECT_ROOT = Path(
    __file__
).parent.parent.parent  # Go up from src/data/ to project root
DEFAULT_FYO_PATH = str(_PROJECT_ROOT / "data" / "FYO")
DEFAULT_CACHE_DIR = str(_PROJECT_ROOT / "data" / "cache")


class FYOScanner(BaseScanner):
    """Scanner for FYO dataset layout."""

    @staticmethod
    def scan(root: str, body_part: str = "all") -> pd.DataFrame:
        """Scan the FYO dataset layout.

        Checks for 'Generated_Images' (10 samples/session) first, then 'ROI' (1 sample/session).

        Args:
            root: Root directory of the FYO dataset.
            body_part: Which body part to scan: "dorsal", "palm", "wrist", or "all".

        Returns:
            DataFrame with columns: ['path', 'patient_id', 'body_part', 'side',
                                    'session', 'finger_class_id', 'dataset'].
        """
        rows = []
        root = Path(root)
        
        # Check for Generated_Images (preferred due to more samples)
        gen_dir = root / "Generated_Images"
        roi_dir = root / "ROI"
        
        if gen_dir.exists():
            base_dir = gen_dir
            mode = "generated"
        elif roi_dir.exists():
            base_dir = roi_dir
            mode = "roi"
        else:
            base_dir = root
            mode = "roi"

        # Determine which body parts to scan
        body_parts_to_scan = {
            "all": ["Dorsal", "Palm", "Wrist"],
            "dorsal": ["Dorsal"],
            "palm": ["Palm"],
            "wrist": ["Wrist"],
        }

        parts = body_parts_to_scan.get(body_part.lower(), ["Dorsal"])

        # Iterate through sessions
        for session_dir in sorted(base_dir.iterdir()):
            if not session_dir.is_dir():
                continue

            session_name = session_dir.name  # e.g., "Session1", "Session2"
            if not session_name.startswith("Session"):
                continue

            # For each body part
            for part in parts:
                part_dir = session_dir / part
                if not part_dir.exists():
                    continue

                # Scan images in body_part directory
                for img_file in part_dir.iterdir():
                    if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
                        continue

                    if mode == "generated":
                        # Naming: s{patient}_{sample}_{side}_S{session}.jpg
                        # e.g. s100_10_L_S1.jpg
                        match = re.match(r"s(\d+)_(\d+)_([LR])_S\d+\.(?:jpg|png|bmp)", img_file.name, re.IGNORECASE)
                        if match:
                            patient_id = match.group(1).zfill(3)
                            side = match.group(3).upper()
                        else:
                            continue
                    else:
                        # ROI Naming: {patient_id}_{side}.png (e.g., "100_L.png")
                        match = re.match(r"(\d+)_([LR])\.png", img_file.name, re.IGNORECASE)
                        if match:
                            patient_id = match.group(1).zfill(3)
                            side = match.group(2).upper()
                        else:
                            continue

                    # Create unique finger class ID
                    finger_class_id = f"fyo_{patient_id}_{part.lower()}_{side}"

                    rows.append(
                        {
                            "path": str(img_file),
                            "patient_id": patient_id,
                            "body_part": part.lower(),
                            "side": side,
                            "session": session_name,
                            "finger_class_id": finger_class_id,
                            "dataset": "fyo",
                        }
                    )

        return pd.DataFrame(rows)


class FYODataset(BaseDataset):
    """Dataset class for FYO multi-body-part vascular images."""

    def __init__(
        self,
        body_part: str = "dorsal",
        df=None,
        transform=None,
        label_encoder=None,
        cfg=None,
    ):
        """Initialize FYO dataset.

        Args:
            body_part: Body part to use: "dorsal", "palm", "wrist", or "all".
            df: Optional DataFrame. If None, will auto-load from DEFAULT_FYO_PATH.
            transform: Optional image transform.
            label_encoder: Optional label encoder.
            cfg: Optional config object. If provided, will auto-build transform.
        """
        if df is None:
            df = build_fyo_manifest(body_part=body_part)

        # Auto-build transform from config if provided
        if cfg is not None and transform is None:
            from .transforms import build_transforms_from_config

            transform = build_transforms_from_config(cfg)

        super().__init__(df, transform, label_encoder)
        self.body_part = body_part

    def _get_metadata(self, row):
        """Extract FYO-specific metadata from DataFrame row.

        Args:
            row: Pandas Series representing one sample.

        Returns:
            dict: Metadata with body part and session information.
        """
        metadata = {
            "finger_class_id": row["finger_class_id"],
            "patient_id": row["patient_id"],
            "body_part": row["body_part"],
            "side": row["side"],
            "session": row["session"],
            "dataset": row.get("dataset", "fyo"),
            "path": row["path"],
        }

        # Add openset_split if available
        if "openset_split" in row:
            metadata["openset_split"] = row["openset_split"]

        # Add sample_split if available (for enrollment/test)
        if "sample_split" in row:
            metadata["sample_split"] = row["sample_split"]

        return metadata

    def get_body_parts(self):
        """Get unique body parts in the dataset."""
        if "body_part" in self.df.columns:
            return self.df["body_part"].unique().tolist()
        return []

    def get_sessions(self):
        """Get unique sessions in the dataset."""
        if "session" in self.df.columns:
            return self.df["session"].unique().tolist()
        return []

    def get_samples_by_body_part(self, body_part: str):
        """Get all samples for a specific body part."""
        if "body_part" not in self.df.columns:
            raise ValueError("Dataset does not have 'body_part' column")
        return self.df[self.df["body_part"] == body_part.lower()]

    def get_samples_by_session(self, session: str):
        """Get all samples for a specific session."""
        if "session" not in self.df.columns:
            raise ValueError("Dataset does not have 'session' column")
        return self.df[self.df["session"] == session]


def build_fyo_manifest(
    body_part: str = "all", cache_dir: str = DEFAULT_CACHE_DIR
) -> pd.DataFrame:
    """Build (or load) a cached manifest for FYO dataset from default paths.

    Args:
        body_part: Body part to include: "dorsal", "palm", "wrist", or "all".
        cache_dir: Directory for cached parquet manifests.

    Returns:
        DataFrame with per-image rows and FYO-specific columns.
    """
    # Use body_part in cache filename to distinguish different scans
    cache_name = f"fyo_{body_part.lower()}"

    # Create custom manifest for this body_part
    cache_path = Path(cache_dir) / f"{cache_name}.parquet"
    Path(cache_dir).mkdir(exist_ok=True)

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = FYOScanner.scan(DEFAULT_FYO_PATH, body_part=body_part)
    df.to_parquet(cache_path, index=False)
    return df
