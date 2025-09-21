"""
DataLoader utilities for training, validation, and testing.

Provides functions to create PyTorch DataLoaders with proper splits and sampling.
"""
from typing import Dict, Tuple

import pandas as pd
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
