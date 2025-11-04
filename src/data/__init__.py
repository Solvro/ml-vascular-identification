"""
Data module for vascular datasets.

This module provides classes for handling dorsal and MMCBNU vascular datasets,
with support for both closed-set and open-set recognition.
"""
from .base import (
    BaseDataset,
    BaseScanner,
    build_manifest_cache,
    create_dataset_from_config,
)
from .data_loaders import (
    create_data_loaders,
    create_data_loaders_from_config,
    create_openset_data_loaders,
    create_single_data_loader,
    get_dorsal_loaders,
    get_mmcbnu_loaders,
)
from .dorsal import (
    DEFAULT_DORSAL_PATH,
    DorsalDataset,
    DorsalScanner,
    build_dorsal_manifest,
)
from .mmcbnu import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MMCBNU_PATH,
    MMCBNUDataset,
    MMCBNUScanner,
    build_mmcbnu_manifest,
)
from .samplers import BalancedBatchSampler
from .splits import (
    make_finger_class_split,
    make_patient_split,
    make_session_split,
    verify_subject_disjoint,
)
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_grayscale_transforms,
    build_transforms,
    build_transforms_from_config,
)

__all__ = [
    # Base classes
    "BaseScanner",
    "BaseDataset",
    "build_manifest_cache",
    "create_dataset_from_config",
    # DataLoaders (Closed-set)
    "create_data_loaders",
    "create_data_loaders_from_config",
    "create_single_data_loader",
    "get_dorsal_loaders",
    "get_mmcbnu_loaders",
    # DataLoaders (OpenSet)
    "create_openset_data_loaders",
    # Dorsal dataset
    "DorsalScanner",
    "DorsalDataset",
    "build_dorsal_manifest",
    "DEFAULT_DORSAL_PATH",
    # MMCBNU dataset
    "MMCBNUScanner",
    "MMCBNUDataset",
    "build_mmcbnu_manifest",
    "DEFAULT_MMCBNU_PATH",
    "DEFAULT_CACHE_DIR",
    # Data utilities
    "make_patient_split",
    "make_finger_class_split",
    "make_session_split",
    "verify_subject_disjoint",
    "build_transforms",
    "build_grayscale_transforms",
    "build_transforms_from_config",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "BalancedBatchSampler",
]
