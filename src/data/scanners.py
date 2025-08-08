import os
import re
import pandas as pd
from pathlib import Path

def scan_dorsal(root: str) -> pd.DataFrame:
    """Scan the 'dorsal' dataset layout.

    Expects P### folders with 'Left'/'Right' subfolders containing images.
    Returns a DataFrame with columns: ['path', 'patient_id', 'side', 'dataset'].
    """
    rows = []
    root = Path(root)
    p_pat = re.compile(r"P(\d+)", re.IGNORECASE)    # match patient folder names like 'P001'
    
    for p_dir in root.iterdir():
        if not p_dir.is_dir(): continue # skip files
        m = p_pat.match(p_dir.name)
        if not m: continue  # skip non-patient dirs
        pid = m.group(1).zfill(3)
        
        for side in ["Left", "Right"]:
            sdir = p_dir / side
            if not sdir.exists(): continue
            for f in sdir.iterdir():
                if f.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]: continue
                rows.append({"path": str(f), "patient_id": pid, "side": side, "dataset": "dorsal"})
                
    return pd.DataFrame(rows)

def scan_mmcbnu(root: str) -> pd.DataFrame:
    """Scan the 'mmcbnu' dataset layout.

    Expects per-patient folders, each containing per-finger subfolders with images.
    Returns a DataFrame with columns: ['path', 'patient_id', 'finger', 'dataset'].
    """
    rows = []
    root = Path(root) 
    
    for pid_dir in root.iterdir():
        if not pid_dir.is_dir(): continue
        pid = pid_dir.name.zfill(3)
        
        for finger_dir in pid_dir.iterdir():
            if not finger_dir.is_dir(): continue
            for f in finger_dir.iterdir():
                if f.suffix.lower() not in [".bmp", ".png", ".jpg", ".jpeg"]: continue
                rows.append({"path": str(f), "patient_id": pid, "finger": finger_dir.name, "dataset": "mmcbnu"})
                
    return pd.DataFrame(rows)

def build_manifest(name: str, root: str, cache_dir: str = "outputs/manifests") -> pd.DataFrame:
    """Build (or load) a cached manifest for a known dataset.

    Args:
        name: Dataset name ('dorsal' | 'mmcbnu').
        root: Root directory of the dataset.
        cache_dir: Directory for cached parquet manifests.

    Returns:
        DataFrame with per-image rows and dataset-specific columns.

    Notes:
        If a cached parquet exists, it is loaded; otherwise the dataset is scanned
        and saved to parquet for faster subsequent loads.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{name}.parquet")
    
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    
    if name == "dorsal":
        df = scan_dorsal(root)
    elif name == "mmcbnu":
        df = scan_mmcbnu(root)
    else:
        raise ValueError(name)
    
    df.to_parquet(cache_path, index=False)
    return df
