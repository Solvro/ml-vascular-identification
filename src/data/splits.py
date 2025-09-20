"""
Patient-wise data splitting utilities.

Ensures that the same patient doesn't appear in multiple splits (train/val/test).
"""
import numpy as np
import pandas as pd


def make_patient_split(
    df: pd.DataFrame, train=0.7, val=0.15, test=0.15, seed=1337
) -> pd.DataFrame:
    """Assign patient-wise splits and return a copy with a 'split' column.

    Args:
        df: DataFrame with at least 'patient_id' column.
        train: Fraction of patients for training.
        val: Fraction of patients for validation.
        test: Fraction of patients for testing.
        seed: Random seed for reproducible splits.

    Returns:
        DataFrame copy with added 'split' column.

    Note:
        Splits are done by patient, not by sample, to avoid data leakage.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    rng = np.random.default_rng(seed)
    pids = sorted(df["patient_id"].unique().tolist())
    rng.shuffle(pids)  # deterministic shuffle by seed

    n = len(pids)
    n_train = int(n * train)
    n_val = int(n * val)

    train_ids = set(pids[:n_train])
    val_ids = set(pids[n_train : n_train + n_val])
    # test_ids = remaining patients

    def _assign_split(pid):
        if pid in train_ids:
            return "train"
        if pid in val_ids:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = [_assign_split(pid) for pid in df["patient_id"]]

    # Sanity check: no patient appears in multiple splits
    train_pids = set(df[df.split == "train"]["patient_id"].unique())
    val_pids = set(df[df.split == "val"]["patient_id"].unique())
    test_pids = set(df[df.split == "test"]["patient_id"].unique())

    assert train_pids.isdisjoint(val_pids), "Train and val sets overlap!"
    assert train_pids.isdisjoint(test_pids), "Train and test sets overlap!"
    assert val_pids.isdisjoint(test_pids), "Val and test sets overlap!"

    return df
