"""
Data splitting utilities for both closed-set and open-set recognition.

Provides functions for:
- Patient-wise splits (closed-set)
- Finger-class-wise splits with subject-disjoint guarantee (open-set)
- Session-wise splits for enrollment/testing
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


def make_finger_class_split(
    df: pd.DataFrame,
    known_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    subject_disjoint: bool = True,
) -> pd.DataFrame:
    """Assign OpenSet splits: known/unknown classes with subject-disjoint guarantee.

    This is the primary splitting function for OpenSet recognition where we treat
    each finger as a separate class. For MMCBNU: 100 patients × 6 fingers = 600 classes.

    Args:
        df: DataFrame with 'patient_id' and 'finger_class_id' columns.
        known_ratio: Fraction of patients (or finger classes) for known set (default 0.7).
        val_ratio: Fraction of KNOWN classes to use for validation (default 0.15).
        seed: Random seed for reproducible splits.
        subject_disjoint: If True, all fingers from same patient go to same split.

    Returns:
        DataFrame with added columns:
        - 'openset_split': 'known' or 'unknown'
        - 'split': 'train', 'val', or 'test'

    Split Logic:
        1. Divide patients into known_patients (70%) and unknown_patients (30%)
        2. All 6 fingers from known_patients → known classes
        3. All 6 fingers from unknown_patients → unknown classes
        4. Within known classes: split into train (85%) and val (15%)
        5. All unknown classes → test only

    Example:
        100 patients → 70 known + 30 unknown
        Known: 70 patients × 6 fingers = 420 classes (357 train, 63 val)
        Unknown: 30 patients × 6 fingers = 180 classes (all test)
    """
    assert 0 < known_ratio < 1, "known_ratio must be between 0 and 1"
    assert 0 < val_ratio < 1, "val_ratio must be between 0 and 1"

    rng = np.random.default_rng(seed)
    df = df.copy()

    if subject_disjoint:
        # Split by PATIENTS to ensure subject-disjoint guarantee
        patients = sorted(df["patient_id"].unique().tolist())
        rng.shuffle(patients)

        n_patients = len(patients)
        n_known_patients = int(n_patients * known_ratio)

        known_patients = set(patients[:n_known_patients])
        unknown_patients = set(patients[n_known_patients:])

        # Assign openset_split based on patient
        df["openset_split"] = df["patient_id"].apply(
            lambda pid: "known" if pid in known_patients else "unknown"
        )

        # Within KNOWN patients, split finger classes into train/val
        known_df = df[df["openset_split"] == "known"].copy()
        known_finger_classes = sorted(known_df["finger_class_id"].unique().tolist())
        rng.shuffle(known_finger_classes)

        n_known_classes = len(known_finger_classes)
        n_val_classes = int(n_known_classes * val_ratio)

        train_finger_classes = set(known_finger_classes[n_val_classes:])
        val_finger_classes = set(known_finger_classes[:n_val_classes])

        def assign_split(row):
            if row["openset_split"] == "unknown":
                return "test"
            elif row["finger_class_id"] in train_finger_classes:
                return "train"
            else:
                return "val"

        df["split"] = df.apply(assign_split, axis=1)

    else:
        # Split by FINGER CLASSES directly (not recommended but available)
        finger_classes = sorted(df["finger_class_id"].unique().tolist())
        rng.shuffle(finger_classes)

        n_classes = len(finger_classes)
        n_known_classes = int(n_classes * known_ratio)

        known_classes = set(finger_classes[:n_known_classes])
        unknown_classes = set(finger_classes[n_known_classes:])

        df["openset_split"] = df["finger_class_id"].apply(
            lambda fc: "known" if fc in known_classes else "unknown"
        )

        # Within known classes, split into train/val
        known_finger_classes = list(known_classes)
        rng.shuffle(known_finger_classes)
        n_val = int(len(known_finger_classes) * val_ratio)

        val_classes = set(known_finger_classes[:n_val])

        def assign_split(row):
            if row["openset_split"] == "unknown":
                return "test"
            elif row["finger_class_id"] in val_classes:
                return "val"
            else:
                return "train"

        df["split"] = df.apply(assign_split, axis=1)

    # Validation: ensure no patient overlap between known/unknown (if subject_disjoint)
    if subject_disjoint:
        known_pids = set(df[df["openset_split"] == "known"]["patient_id"].unique())
        unknown_pids = set(df[df["openset_split"] == "unknown"]["patient_id"].unique())
        assert known_pids.isdisjoint(
            unknown_pids
        ), "Subject-disjoint constraint violated!"

    # Validation: ensure train/val only contain known classes
    assert (
        df[df["split"] == "train"]["openset_split"] == "known"
    ).all(), "Train must only contain known classes!"
    assert (
        df[df["split"] == "val"]["openset_split"] == "known"
    ).all(), "Val must only contain known classes!"
    assert (
        df[df["split"] == "test"]["openset_split"] == "unknown"
    ).all(), "Test must only contain unknown classes!"

    return df


def make_session_split(
    df: pd.DataFrame,
    enrollment_samples: int = 7,
    test_samples: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Split samples within each finger class into enrollment and test sets.

    This is useful for:
    - Creating prototypes/centroids from enrollment samples
    - Testing on held-out samples of the same finger class
    - Simulating real-world enrollment → verification workflow

    Args:
        df: DataFrame with 'finger_class_id' column.
        enrollment_samples: Number of samples per finger for enrollment (default 7).
        test_samples: Number of samples per finger for testing (default 3).
        seed: Random seed for reproducible splits.

    Returns:
        DataFrame with added 'sample_split' column: 'enrollment' or 'test'.

    Note:
        For MMCBNU, each finger has 10 samples, so 7/3 split is reasonable.
        If a finger has fewer samples, all go to enrollment.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    sample_splits = []

    for finger_class_id in df["finger_class_id"].unique():
        finger_df = df[df["finger_class_id"] == finger_class_id]
        indices = finger_df.index.tolist()

        # Shuffle indices deterministically
        indices_array = np.array(indices)
        rng.shuffle(indices_array)
        indices = indices_array.tolist()

        n_samples = len(indices)
        n_enrollment = min(enrollment_samples, n_samples)

        enrollment_indices = set(indices[:n_enrollment])

        for idx in indices:
            if idx in enrollment_indices:
                sample_splits.append((idx, "enrollment"))
            else:
                sample_splits.append((idx, "test"))

    # Create a mapping
    split_map = {idx: split for idx, split in sample_splits}
    df["sample_split"] = df.index.map(split_map)

    return df


def verify_subject_disjoint(df: pd.DataFrame) -> dict:
    """Verify subject-disjoint constraint and return split statistics.

    Args:
        df: DataFrame with 'patient_id', 'openset_split', and 'split' columns.

    Returns:
        Dictionary with detailed statistics about the splits.

    Raises:
        AssertionError: If subject-disjoint constraint is violated.
    """
    if "openset_split" not in df.columns:
        return {"error": "DataFrame missing 'openset_split' column"}

    known_df = df[df["openset_split"] == "known"]
    unknown_df = df[df["openset_split"] == "unknown"]

    known_patients = set(known_df["patient_id"].unique())
    unknown_patients = set(unknown_df["patient_id"].unique())

    # Check for overlap
    overlap = known_patients & unknown_patients
    if overlap:
        raise AssertionError(
            f"Subject-disjoint constraint violated! Overlapping patients: {overlap}"
        )

    # Collect statistics
    stats = {
        "subject_disjoint": True,
        "total_patients": len(df["patient_id"].unique()),
        "known_patients": len(known_patients),
        "unknown_patients": len(unknown_patients),
        "total_finger_classes": df["finger_class_id"].nunique(),
        "known_finger_classes": known_df["finger_class_id"].nunique(),
        "unknown_finger_classes": unknown_df["finger_class_id"].nunique(),
        "splits": {},
    }

    # Per-split statistics
    for split_name in df["split"].unique():
        split_df = df[df["split"] == split_name]
        stats["splits"][split_name] = {
            "samples": len(split_df),
            "patients": split_df["patient_id"].nunique(),
            "finger_classes": split_df["finger_class_id"].nunique(),
            "openset_split": split_df["openset_split"].iloc[0]
            if len(split_df) > 0
            else None,
        }

    return stats
