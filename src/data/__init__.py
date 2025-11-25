"""
Data module for vascular datasets.

This module provides classes for handling dorsal, MMCBNU, FYO, and UTFVP vascular datasets,
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
    get_fyo_loaders,
    get_mmcbnu_loaders,
    get_utfvp_loaders,
)
from .dorsal import (
    DEFAULT_DORSAL_PATH,
    DorsalDataset,
    DorsalScanner,
    build_dorsal_manifest,
)
from .fyo import (
    DEFAULT_FYO_PATH,
    FYODataset,
    FYOScanner,
    build_fyo_manifest,
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
    extract_roi_dorsal,
)
from .utfvp import (
    DEFAULT_UTFVP_PATH,
    UTFVPDataset,
    UTFVPScanner,
    build_utfvp_manifest,
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
    "get_fyo_loaders",
    "get_utfvp_loaders",
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
    # FYO dataset
    "FYOScanner",
    "FYODataset",
    "build_fyo_manifest",
    "DEFAULT_FYO_PATH",
    # UTFVP dataset
    "UTFVPScanner",
    "UTFVPDataset",
    "build_utfvp_manifest",
    "DEFAULT_UTFVP_PATH",
    # Cache
    "DEFAULT_CACHE_DIR",
    # Data utilities
    "make_patient_split",
    "make_finger_class_split",
    "make_session_split",
    "verify_subject_disjoint",
    "build_transforms",
    "build_grayscale_transforms",
    "build_transforms_from_config",
    "extract_roi_dorsal",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "BalancedBatchSampler",
]
