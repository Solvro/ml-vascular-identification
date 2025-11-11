"""
Image transformation utilities for vascular datasets.

Provides standardized augmentations and normalization for training and evaluation,
with optional ROI extraction for dorsal vein dataset.
"""
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ImageNet statistics (commonly used for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_roi_dorsal(
    img: Image.Image, method: str = "otsu", padding: int = 10
) -> Image.Image:
    """Extract ROI (Region of Interest) from dorsal hand vein image.

    This function isolates the vascular pattern from the background for improved
    training and inference. Converts PIL Image → OpenCV → extract ROI → back to PIL.

    Args:
        img: PIL Image in grayscale mode.
        method: ROI extraction method:
            - "otsu": Otsu's thresholding (robust, no params needed)
            - "adaptive": Adaptive thresholding (good for varying lighting)
            - "percentile": Threshold at N-th percentile of pixel intensity
        padding: Pixels to add around detected ROI boundary (default 10).

    Returns:
        PIL Image (grayscale) with ROI extracted and padded.

    Note:
        If ROI detection fails or covers entire image, returns original image (as grayscale).
    """
    # Convert PIL → grayscale if needed
    if img.mode != "L":
        img = img.convert("L")

    # Convert PIL → numpy array (grayscale)
    img_array = np.array(img)
    h, w = img_array.shape

    # Threshold to get binary mask of vascular pattern
    if method == "otsu":
        _, binary = cv2.threshold(
            img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "percentile":
        threshold_val = np.percentile(img_array, 30)  # Bottom 30% as foreground
        binary = (img_array < threshold_val).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown ROI extraction method: {method}")

    # Find contours and get bounding box
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No vascular pattern detected, return original (as grayscale)
        return img

    # Get largest contour (vascular region)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(largest_contour)

    # Check if ROI is reasonable (not covering entire image)
    roi_area = (box_w * box_h) / (h * w)
    if roi_area > 0.95:
        # ROI covers >95% of image, not useful - return original (as grayscale)
        return img

    # Apply padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + box_w + padding)
    y2 = min(h, y + box_h + padding)

    # Crop ROI
    roi_array = img_array[y1:y2, x1:x2]

    # Convert back to PIL Image (grayscale)
    return Image.fromarray(roi_array)


def build_transforms(
    img_size: int = 256,
    train: bool = True,
    hflip_p: float = 0.5,
    roi_extraction: bool = False,
    roi_method: str = "otsu",
) -> transforms.Compose:
    """Build torchvision transform pipeline for train or eval.

    Args:
        img_size: Target image size (square).
        train: If True, apply data augmentations.
        hflip_p: Probability of horizontal flip (only for training).
        roi_extraction: If True, apply ROI extraction before resizing (for dorsal dataset).
        roi_method: ROI extraction method ('otsu', 'adaptive', 'percentile').

    Returns:
        torchvision.transforms.Compose object.

    Note:
        Converts grayscale to RGB (3-channel) for compatibility with
        pre-trained models that expect RGB input.
    """
    pipeline = []

    # ROI extraction (before any transforms)
    if roi_extraction:
        pipeline.append(
            transforms.Lambda(lambda img: extract_roi_dorsal(img, method=roi_method))
        )

    # Convert grayscale to RGB
    pipeline.append(transforms.Grayscale(num_output_channels=3))

    # Resize
    pipeline.append(transforms.Resize((img_size, img_size)))

    # Augmentations (training only)
    if train:
        pipeline.append(transforms.RandomHorizontalFlip(hflip_p))

    # Tensor conversion and normalization
    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

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

    # ROI extraction (only for dorsal)
    roi_extraction = False
    roi_method = "otsu"
    if dataset_name == "dorsal":
        roi_extraction = getattr(t_cfg, "roi_extraction", False)
        roi_method = getattr(t_cfg, "roi_method", "otsu")

    if normalize == "imagenet":
        return build_transforms(
            img_size,
            train,
            hflip_p,
            roi_extraction=roi_extraction,
            roi_method=roi_method,
        )
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
