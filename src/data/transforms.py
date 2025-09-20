"""
Image transformation utilities for vascular datasets.

Provides standardized augmentations and normalization for training and evaluation.
"""
from torchvision import transforms

# ImageNet statistics (commonly used for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size=256, train=True, hflip_p=0.5):
    """Build torchvision transform pipeline for train or eval.

    Args:
        img_size: Target image size (square).
        train: If True, apply data augmentations.
        hflip_p: Probability of horizontal flip (only for training).

    Returns:
        torchvision.transforms.Compose object.

    Note:
        Converts grayscale to RGB (3-channel) for compatibility with
        pre-trained models that expect RGB input.
    """
    if train:
        pipeline = [
            transforms.Grayscale(num_output_channels=3),  # L→RGB (3ch)
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(hflip_p),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    else:
        pipeline = [
            transforms.Grayscale(num_output_channels=3),  # L→RGB (3ch)
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

    return transforms.Compose(pipeline)


def build_transforms_from_config(cfg, dataset_name=None):
    """Build transforms from config object.

    Args:
        cfg: Config object with hierarchical structure:
            cfg.data.dorsal.transforms or cfg.data.mmcbnu.transforms
        dataset_name: Optional dataset name ('dorsal' | 'mmcbnu').
                     If None, will use cfg.data.dataset.

    Returns:
        torchvision.transforms.Compose object or None.
    """
    if not hasattr(cfg, "data"):
        return None

    # Determine dataset name
    if dataset_name is None:
        dataset_name = getattr(cfg.data, "dataset", None)

    if dataset_name is None:
        return None

    # Get dataset-specific config
    if not hasattr(cfg.data, dataset_name):
        return None

    dataset_cfg = getattr(cfg.data, dataset_name)

    if not hasattr(dataset_cfg, "transforms"):
        return None

    t_cfg = dataset_cfg.transforms
    img_size = getattr(t_cfg, "img_size", 256)
    train = getattr(t_cfg, "train", True)
    hflip_p = getattr(t_cfg, "hflip_p", 0.5)
    normalize = getattr(t_cfg, "normalize", "imagenet")

    if normalize == "imagenet":
        return build_transforms(img_size, train, hflip_p)
    elif normalize == "none":
        return build_grayscale_transforms(img_size, train, hflip_p)
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")


def get_split_config(cfg, dataset_name=None):
    """Get split configuration for specific dataset.

    Args:
        cfg: Config object.
        dataset_name: Dataset name ('dorsal' | 'mmcbnu').

    Returns:
        Split config object or None.
    """
    if not hasattr(cfg, "data"):
        return None

    if dataset_name is None:
        dataset_name = getattr(cfg.data, "dataset", None)

    if dataset_name is None or not hasattr(cfg.data, dataset_name):
        return None

    dataset_cfg = getattr(cfg.data, dataset_name)
    return getattr(dataset_cfg, "split", None)


def build_grayscale_transforms(img_size=256, train=True, hflip_p=0.5):
    """Build transform pipeline for grayscale output (single channel).

    Args:
        img_size: Target image size (square).
        train: If True, apply data augmentations.
        hflip_p: Probability of horizontal flip (only for training).

    Returns:
        torchvision.transforms.Compose object.
    """
    if train:
        pipeline = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(hflip_p),
            transforms.ToTensor(),
        ]
    else:
        pipeline = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]

    return transforms.Compose(pipeline)
