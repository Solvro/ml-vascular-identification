import numpy as np
import pandas as pd

def make_patient_split(df: pd.DataFrame, train=0.7, val=0.15, test=0.15, seed=1337) -> pd.DataFrame:
    """Assign patient-wise splits and return a copy with a 'split' column."""
    assert abs(train + val + test - 1.0) < 1e-6
    
    rng = np.random.default_rng(seed)
    pids = sorted(df["patient_id"].unique().tolist())
    rng.shuffle(pids)   # deterministic shuffle by seed
    
    n = len(pids)
    n_train = int(n * train)
    n_val = int(n * val)
    
    train_ids = set(pids[:n_train])
    val_ids = set(pids[n_train:n_train + n_val])
    
    def _split(pid):
        if pid in train_ids: return "train"
        if pid in val_ids: return "val"
        return "test"
    
    df = df.copy()
    df["split"] = [ _split(pid) for pid in df["patient_id"] ]
    return df
