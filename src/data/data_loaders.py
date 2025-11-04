"""
DataLoader utilities for training, validation, and testing.

Provides functions to create PyTorch DataLoaders with proper splits and sampling.
"""
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from .base import BaseDataset
from .dorsal import DorsalDataset
from .mmcbnu import MMCBNUDataset
from .samplers import BalancedBatchSampler
from .splits import make_patient_split
from .transforms import build_transforms


def create_dataset_from_name(
    name: str, df: pd.DataFrame, transform=None, label_encoder=None
) -> BaseDataset:
    """Create dataset instance from name.

    Args:
        name: Dataset name ('dorsal' or 'mmcbnu').
        df: DataFrame with samples.
        transform: Optional transform.
        label_encoder: Optional label encoder.

    Returns:
        Dataset instance.
    """
    if name == "dorsal":
        return DorsalDataset(df=df, transform=transform, label_encoder=label_encoder)
    elif name == "mmcbnu":
        return MMCBNUDataset(df=df, transform=transform, label_encoder=label_encoder)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def create_data_loaders_from_config(
    cfg
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create DataLoaders from Hydra config object.

    Args:
        cfg: Config object with data configuration.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict).
    """
    # Extract config values with defaults
    # Dataset name and data config (from root level due to @package _global_)
    dataset_name = cfg.name
    img_size = getattr(cfg.transforms, "img_size", 256)
    hflip_p = getattr(cfg.transforms, "hflip_p", 0.5)

    # Split configuration (from root level due to @package _global_)
    train_split = getattr(cfg.split, "train", 0.7)
    val_split = getattr(cfg.split, "val", 0.15)
    test_split = getattr(cfg.split, "test", 0.15)
    split_seed = getattr(cfg.split, "seed", 42)

    # Sampling configuration (from loader module)
    P = (
        getattr(cfg.loader.sampler, "P", 8)
        if hasattr(cfg, "loader") and hasattr(cfg.loader, "sampler")
        else 8
    )
    K = (
        getattr(cfg.loader.sampler, "K", 4)
        if hasattr(cfg, "loader") and hasattr(cfg.loader, "sampler")
        else 4
    )

    # DataLoader configuration (from loader module)
    batch_size = getattr(cfg.loader, "batch_size", 64) if hasattr(cfg, "loader") else 64
    num_workers = getattr(cfg.loader, "num_workers", 4) if hasattr(cfg, "loader") else 4
    pin_memory = (
        getattr(cfg.loader, "pin_memory", True) if hasattr(cfg, "loader") else True
    )

    return create_data_loaders(
        dataset_name=dataset_name,
        img_size=img_size,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        split_seed=split_seed,
        P=P,
        K=K,
        num_workers=num_workers,
        pin_memory=pin_memory,
        batch_size=batch_size,
        hflip_p=hflip_p,
    )


def create_data_loaders(
    dataset_name: str,
    img_size: int = 256,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    split_seed: int = 42,
    # Training DataLoader params
    P: int = 8,  # Classes per batch
    K: int = 4,  # Samples per class
    num_workers: int = 4,
    pin_memory: bool = True,
    # Validation/Test DataLoader params
    batch_size: int = 64,
    # Transform params
    hflip_p: float = 0.5,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create train, validation, and test DataLoaders.

    Args:
        dataset_name: Name of dataset ('dorsal' or 'mmcbnu').
        img_size: Image size for transforms.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        test_split: Fraction for testing.
        split_seed: Random seed for reproducible splits.
        P: Number of classes per batch (for training).
        K: Number of samples per class (for training).
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory.
        batch_size: Batch size for val/test loaders.
        hflip_p: Horizontal flip probability for training.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict).
        info_dict contains dataset statistics and label encoders.
    """
    # Load full dataset
    if dataset_name == "dorsal":
        full_dataset = DorsalDataset()
    elif dataset_name == "mmcbnu":
        full_dataset = MMCBNUDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split by patients
    df_with_splits = make_patient_split(
        full_dataset.df,
        train=train_split,
        val=val_split,
        test=test_split,
        seed=split_seed,
    )

    # Create DataFrames for each split
    train_df = (
        df_with_splits[df_with_splits.split == "train"].copy().reset_index(drop=True)
    )
    val_df = df_with_splits[df_with_splits.split == "val"].copy().reset_index(drop=True)
    test_df = (
        df_with_splits[df_with_splits.split == "test"].copy().reset_index(drop=True)
    )

    # Create transforms
    train_transform = build_transforms(img_size=img_size, train=True, hflip_p=hflip_p)
    eval_transform = build_transforms(img_size=img_size, train=False)

    # Create datasets for each split
    train_dataset = create_dataset_from_name(
        dataset_name, train_df, transform=train_transform
    )

    # Use train label encoder for all splits to ensure consistency
    # But add any missing patient IDs from val/test
    all_patient_ids = sorted(df_with_splits["patient_id"].unique())
    global_label_encoder = {pid: i for i, pid in enumerate(all_patient_ids)}

    # Recreate train dataset with global encoder
    train_dataset = create_dataset_from_name(
        dataset_name,
        train_df,
        transform=train_transform,
        label_encoder=global_label_encoder,
    )
    val_dataset = create_dataset_from_name(
        dataset_name,
        val_df,
        transform=eval_transform,
        label_encoder=global_label_encoder,
    )
    test_dataset = create_dataset_from_name(
        dataset_name,
        test_df,
        transform=eval_transform,
        label_encoder=global_label_encoder,
    )

    # Create balanced batch sampler for training
    train_labels = [train_dataset.le[pid] for pid in train_df["patient_id"]]
    train_sampler = BalancedBatchSampler(labels=train_labels, P=P, K=K)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Collect statistics
    sampler_stats = train_sampler.get_stats()
    info = {
        "dataset_name": dataset_name,
        "splits": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "train_patients": len(train_df.patient_id.unique()),
            "val_patients": len(val_df.patient_id.unique()),
            "test_patients": len(test_df.patient_id.unique()),
        },
        "sampling": {
            "P": P,
            "K": K,
            "batch_size": P * K,
            "batches_per_epoch": sampler_stats["batches_per_epoch"],
            "total_classes": sampler_stats["total_classes"],
        },
        "label_encoder": global_label_encoder,
        "transforms": {
            "img_size": img_size,
            "hflip_p": hflip_p,
        },
    }

    return train_loader, val_loader, test_loader, info


def create_single_data_loader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    img_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs,
) -> Tuple[DataLoader, Dict]:
    """Create a single DataLoader for quick testing or inference.

    Args:
        dataset_name: Name of dataset ('dorsal' or 'mmcbnu').
        split: Which split to create ('train', 'val', 'test', or 'full').
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        img_size: Image size.
        num_workers: Number of workers.
        pin_memory: Whether to pin memory.
        **kwargs: Additional arguments for make_patient_split.

    Returns:
        Tuple of (data_loader, info_dict).
    """
    # Load dataset
    if dataset_name == "dorsal":
        dataset = DorsalDataset()
    elif dataset_name == "mmcbnu":
        dataset = MMCBNUDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Apply split if needed
    if split != "full":
        df_with_splits = make_patient_split(dataset.df, **kwargs)
        df = df_with_splits[df_with_splits.split == split].copy().reset_index(drop=True)
        dataset = create_dataset_from_name(
            dataset_name,
            df,
            transform=build_transforms(img_size=img_size, train=(split == "train")),
        )

    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Info
    info = {
        "dataset_name": dataset_name,
        "split": split,
        "num_samples": len(dataset),
        "num_patients": len(dataset.get_patient_ids()),
        "batch_size": batch_size,
        "num_batches": len(data_loader),
    }

    return data_loader, info


# Convenience function for quick access
def get_dorsal_loaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Get dorsal train/val/test loaders with default parameters."""
    return create_data_loaders("dorsal", **kwargs)


def get_mmcbnu_loaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Get MMCBNU train/val/test loaders with default parameters."""
    return create_data_loaders("mmcbnu", **kwargs)


def create_openset_data_loaders(
    dataset_name: str = "mmcbnu",
    img_size: int = 224,
    known_ratio: float = 0.7,
    val_ratio: float = 0.15,
    subject_disjoint: bool = True,
    enrollment_samples: int = 7,
    test_samples: int = 3,
    seed: int = 42,
    # Training DataLoader params
    P: int = 16,
    K: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    # Validation/Test DataLoader params
    batch_size: int = 64,
    # Transform params
    hflip_p: float = 0.3,
) -> Tuple[Dict[str, DataLoader], Dict]:
    """Create DataLoaders for OpenSet Recognition with subject-disjoint splits.

    This is the primary function for creating dataloaders in OpenSet mode where:
    - Each finger is treated as a separate class (600 classes for MMCBNU)
    - Known/unknown splits are subject-disjoint (all 6 fingers from same patient in same split)
    - Training uses only known classes with P-K sampling for metric learning
    - Validation uses only known classes for threshold tuning
    - Test uses only unknown classes for open-set evaluation

    Args:
        dataset_name: Name of dataset ('mmcbnu' for now, 'dorsal' later).
        img_size: Image size for transforms.
        known_ratio: Fraction of patients for known classes (default 0.7).
        val_ratio: Fraction of KNOWN classes for validation (default 0.15).
        subject_disjoint: Ensure all fingers from same patient in same split.
        enrollment_samples: Samples per finger class for enrollment/prototypes (default 7).
        test_samples: Samples per finger class for testing (default 3).
        seed: Random seed for reproducible splits.
        P: Number of classes per batch for training (default 16).
        K: Number of samples per class for training (default 4).
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory.
        batch_size: Batch size for val/test loaders.
        hflip_p: Horizontal flip probability for training.

    Returns:
        Tuple of (loaders_dict, info_dict) where:
        - loaders_dict contains:
            'train': DataLoader for known classes training
            'val_known': DataLoader for known classes validation
            'test_known': DataLoader for known classes testing (enrollment samples)
            'test_unknown': DataLoader for unknown classes testing
        - info_dict contains comprehensive statistics and metadata.

    Example:
        >>> loaders, info = create_openset_data_loaders('mmcbnu', P=16, K=4)
        >>> train_loader = loaders['train']
        >>> test_unknown_loader = loaders['test_unknown']
        >>> print(f"Known classes: {info['known_finger_classes']}")
    """
    from .splits import make_finger_class_split, make_session_split, verify_subject_disjoint

    # Load full dataset and create subject-level openset splits.
    if dataset_name == "mmcbnu":
        # Existing MMCBNU flow (finger-class based)
        full_dataset = MMCBNUDataset()

        # Step 1: Apply finger-class-level OpenSet split (known/unknown)
        df_with_openset = make_finger_class_split(
            full_dataset.df,
            known_ratio=known_ratio,
            val_ratio=val_ratio,
            seed=seed,
            subject_disjoint=subject_disjoint,
        )

        # Verify subject-disjoint constraint
        split_stats = verify_subject_disjoint(df_with_openset)

        # Step 2: Apply session split (enrollment/test samples within each finger)
        df_complete = make_session_split(
            df_with_openset,
            enrollment_samples=enrollment_samples,
            test_samples=test_samples,
            seed=seed,
        )
    elif dataset_name == "dorsal":
        # Dorsal: hand-level classes (Left/Right). Use subject-level partition as requested:
        # 55% TrainKnown, 10% ValKnown, 5% TestKnown, 30% Unknown (by subject)
        full_dataset = DorsalDataset()
        df = full_dataset.df.copy()

        # Get unique patients and shuffle
        patients = sorted(df["patient_id"].unique().tolist())
        rng = np.random.default_rng(seed)
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(n * 0.55)
        n_val = int(n * 0.10)
        n_test_known = int(n * 0.05)
        # Rest become unknown
        n_assigned = n_train + n_val + n_test_known
        n_unknown = max(0, n - n_assigned)

        # Assign patient groups
        train_patients = set(patients[:n_train])
        val_patients = set(patients[n_train : n_train + n_val])
        test_known_patients = set(patients[n_train + n_val : n_train + n_val + n_test_known])
        unknown_patients = set(patients[n_train + n_val + n_test_known :])

        # Create openset_split and split columns (subject-disjoint)
        def _assign_patient_split(pid):
            if pid in unknown_patients:
                return ("unknown", "test")
            elif pid in test_known_patients:
                return ("known", "test")
            elif pid in val_patients:
                return ("known", "val")
            else:
                return ("known", "train")

        openset_splits = []
        splits = []
        for pid in df["patient_id"]:
            o, s = _assign_patient_split(pid)
            openset_splits.append(o)
            splits.append(s)

        df["openset_split"] = openset_splits
        df["split"] = splits

        # Verify subject-disjoint: patients assigned to one group only
        split_stats = verify_subject_disjoint(df)

        # Step 2: Create enrollment/test sample splits per hand (finger_class_id expected in dorsal scanner)
        df_complete = make_session_split(
            df,
            enrollment_samples=enrollment_samples,
            test_samples=test_samples,
            seed=seed,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet. Use 'mmcbnu' or 'dorsal'.")

    # Step 3: Create DataFrames for each split
    # Train: known classes, all samples (or enrollment samples)
    train_df = (
        df_complete[(df_complete["split"] == "train")]
        .copy()
        .reset_index(drop=True)
    )

    # Val: known classes, all samples
    val_known_df = (
        df_complete[(df_complete["split"] == "val")]
        .copy()
        .reset_index(drop=True)
    )

    # Test known enrollment: known classes, enrollment samples (for prototypes)
    test_known_enrollment_df = (
        df_complete[
            (df_complete["openset_split"] == "known")
            & (df_complete["sample_split"] == "enrollment")
        ]
        .copy()
        .reset_index(drop=True)
    )
    
    # Test known query: known classes, test samples (for querying against prototypes)
    test_known_query_df = (
        df_complete[
            (df_complete["openset_split"] == "known")
            & (df_complete["sample_split"] == "test")
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Test unknown: unknown classes, all samples
    test_unknown_df = (
        df_complete[(df_complete["split"] == "test")]
        .copy()
        .reset_index(drop=True)
    )

    # Step 4: Create transforms
    train_transform = build_transforms(img_size=img_size, train=True, hflip_p=hflip_p)
    eval_transform = build_transforms(img_size=img_size, train=False)

    # Step 5: Build global label encoder for ALL finger classes
    all_finger_classes = sorted(df_complete["finger_class_id"].unique())
    global_label_encoder = {fc: i for i, fc in enumerate(all_finger_classes)}

    # Step 6: Create datasets with use_finger_classes=True
    train_dataset = create_dataset_from_name(
        dataset_name,
        train_df,
        transform=train_transform,
        label_encoder=global_label_encoder,
    )
    train_dataset.use_finger_classes = True
    train_dataset.id_column = "finger_class_id"

    val_known_dataset = create_dataset_from_name(
        dataset_name,
        val_known_df,
        transform=eval_transform,
        label_encoder=global_label_encoder,
    )
    val_known_dataset.use_finger_classes = True
    val_known_dataset.id_column = "finger_class_id"

    # Test known enrollment dataset (for computing prototypes)
    test_known_enrollment_dataset = create_dataset_from_name(
        dataset_name,
        test_known_enrollment_df,
        transform=eval_transform,
        label_encoder=global_label_encoder,
    )
    test_known_enrollment_dataset.use_finger_classes = True
    test_known_enrollment_dataset.id_column = "finger_class_id"
    
    # Test known query dataset (for testing against prototypes)
    test_known_query_dataset = create_dataset_from_name(
        dataset_name,
        test_known_query_df,
        transform=eval_transform,
        label_encoder=global_label_encoder,
    )
    test_known_query_dataset.use_finger_classes = True
    test_known_query_dataset.id_column = "finger_class_id"

    test_unknown_dataset = create_dataset_from_name(
        dataset_name,
        test_unknown_df,
        transform=eval_transform,
        label_encoder=global_label_encoder,
    )
    test_unknown_dataset.use_finger_classes = True
    test_unknown_dataset.id_column = "finger_class_id"

    # Step 7: Create balanced batch sampler for training
    train_labels = [train_dataset.le[fc] for fc in train_df["finger_class_id"]]
    train_sampler = BalancedBatchSampler(labels=train_labels, P=P, K=K)

    # Step 8: Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_known_loader = DataLoader(
        val_known_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    test_known_enrollment_loader = DataLoader(
        test_known_enrollment_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    
    test_known_query_loader = DataLoader(
        test_known_query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    test_unknown_loader = DataLoader(
        test_unknown_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Step 9: Collect comprehensive statistics
    sampler_stats = train_sampler.get_stats()

    loaders = {
        "train": train_loader,
        "val_known": val_known_loader,
        "test_known_enrollment": test_known_enrollment_loader,
        "test_known_query": test_known_query_loader,
        "test_unknown": test_unknown_loader,
    }

    info = {
        "dataset_name": dataset_name,
        "mode": "openset",
        "subject_disjoint": subject_disjoint,
        # OpenSet statistics
        "total_finger_classes": len(all_finger_classes),
        "known_finger_classes": train_df["finger_class_id"].nunique()
        + val_known_df["finger_class_id"].nunique(),
        "unknown_finger_classes": test_unknown_df["finger_class_id"].nunique(),
        "known_patients": sorted(
            df_complete[df_complete["openset_split"] == "known"]["patient_id"]
            .unique()
            .tolist()
        ),
        "unknown_patients": sorted(
            df_complete[df_complete["openset_split"] == "unknown"]["patient_id"]
            .unique()
            .tolist()
        ),
        # Split statistics
        "splits": {
            "train": {
                "samples": len(train_dataset),
                "finger_classes": train_df["finger_class_id"].nunique(),
                "patients": train_df["patient_id"].nunique(),
            },
            "val_known": {
                "samples": len(val_known_dataset),
                "finger_classes": val_known_df["finger_class_id"].nunique(),
                "patients": val_known_df["patient_id"].nunique(),
            },
            "test_known_enrollment": {
                "samples": len(test_known_enrollment_dataset),
                "finger_classes": test_known_enrollment_df["finger_class_id"].nunique(),
                "patients": test_known_enrollment_df["patient_id"].nunique(),
                "note": "enrollment samples for computing prototypes",
            },
            "test_known_query": {
                "samples": len(test_known_query_dataset),
                "finger_classes": test_known_query_df["finger_class_id"].nunique(),
                "patients": test_known_query_df["patient_id"].nunique(),
                "note": "test samples for querying against prototypes",
            },
            "test_unknown": {
                "samples": len(test_unknown_dataset),
                "finger_classes": test_unknown_df["finger_class_id"].nunique(),
                "patients": test_unknown_df["patient_id"].nunique(),
            },
        },
        # Sampling statistics
        "sampling": {
            "P": P,
            "K": K,
            "batch_size": P * K,
            "batches_per_epoch": sampler_stats["batches_per_epoch"],
        },
        # Session split info
        "enrollment": {
            "enrollment_samples": enrollment_samples,
            "test_samples": test_samples,
        },
        # Transform info
        "transforms": {
            "img_size": img_size,
            "hflip_p": hflip_p,
        },
        # Label encoder
        "label_encoder": global_label_encoder,
        # Verification stats
        "verification": split_stats,
    }

    return loaders, info


if __name__ == "__main__":
    # Example usage
    print("🔍 Testing DataLoader creation...")

    try:
        train_loader, val_loader, test_loader, info = get_dorsal_loaders(
            P=4, K=2, img_size=224
        )

        print(f"✅ Created loaders for {info['dataset_name']}:")
        print(f"   📊 Splits: {info['splits']}")
        print(f"   🎯 Sampling: {info['sampling']}")
        print(f"   🖼️  Transforms: {info['transforms']}")

        # Test one batch
        batch = next(iter(train_loader))
        images, labels, metadata = batch
        print(f"   📦 Batch shape: {images.shape}, labels: {len(labels)}")

    except Exception as e:
        print(f"❌ Error: {e}")
