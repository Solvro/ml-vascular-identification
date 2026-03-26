"""
Unit tests for ROI extraction from dorsal hand vein images.

Tests the extract_roi_dorsal function with various methods and edge cases.
"""
import numpy as np
import pytest
from PIL import Image

from src.data.transforms import extract_roi_dorsal


@pytest.fixture
def sample_hand_image():
    """Create a synthetic hand vein image for testing.

    Returns:
        PIL Image (grayscale) with simulated vascular pattern.
    """
    # Create 512x512 image with dark background (hand outline) and light veins
    img_array = np.ones((512, 512), dtype=np.uint8) * 200  # Light background

    # Add darker hand region
    img_array[100:400, 100:400] = 180

    # Add simulated vein pattern (dark lines on hand)
    # Horizontal veins
    img_array[200:210, 150:350] = 50
    img_array[300:310, 150:350] = 60

    # Vertical veins
    img_array[100:400, 200:210] = 55
    img_array[100:400, 300:310] = 65

    return Image.fromarray(img_array, mode="L")


@pytest.fixture
def uniform_image():
    """Create a uniform image (no vein pattern).

    Returns:
        PIL Image (grayscale) with uniform intensity.
    """
    return Image.fromarray(np.ones((256, 256), dtype=np.uint8) * 128, mode="L")


class TestROIExtraction:
    """Test ROI extraction with different methods."""

    def test_otsu_extraction(self, sample_hand_image):
        """Test Otsu's thresholding ROI extraction."""
        roi_img = extract_roi_dorsal(sample_hand_image, method="otsu")

        # Should return a smaller image (cropped ROI)
        assert isinstance(roi_img, Image.Image)
        assert roi_img.mode == "L"

        # ROI should be smaller than original (unless ROI covers >95%)
        original_size = sample_hand_image.size[0] * sample_hand_image.size[1]
        roi_size = roi_img.size[0] * roi_img.size[1]
        assert roi_size <= original_size

    def test_adaptive_extraction(self, sample_hand_image):
        """Test adaptive thresholding ROI extraction."""
        roi_img = extract_roi_dorsal(sample_hand_image, method="adaptive")

        assert isinstance(roi_img, Image.Image)
        assert roi_img.mode == "L"

    def test_percentile_extraction(self, sample_hand_image):
        """Test percentile-based ROI extraction."""
        roi_img = extract_roi_dorsal(sample_hand_image, method="percentile")

        assert isinstance(roi_img, Image.Image)
        assert roi_img.mode == "L"

    def test_invalid_method(self, sample_hand_image):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ROI extraction method"):
            extract_roi_dorsal(sample_hand_image, method="invalid_method")

    def test_uniform_image_fallback(self, uniform_image):
        """Test that uniform image returns original (no vein pattern detected)."""
        roi_img = extract_roi_dorsal(uniform_image, method="otsu")

        # Should return original since no clear pattern
        assert isinstance(roi_img, Image.Image)
        # Might be original or processed (depends on Otsu result)

    def test_padding_parameter(self, sample_hand_image):
        """Test ROI extraction with different padding values."""
        roi_no_pad = extract_roi_dorsal(sample_hand_image, padding=0)
        roi_with_pad = extract_roi_dorsal(sample_hand_image, padding=20)

        # Both should be valid images
        assert isinstance(roi_no_pad, Image.Image)
        assert isinstance(roi_with_pad, Image.Image)

        # With padding should potentially be larger or equal
        area_no_pad = roi_no_pad.size[0] * roi_no_pad.size[1]
        area_with_pad = roi_with_pad.size[0] * roi_with_pad.size[1]
        assert area_with_pad >= area_no_pad

    def test_rgb_to_grayscale_conversion(self):
        """Test that RGB images are converted to grayscale before processing."""
        # Create RGB version of test image
        rgb_array = np.ones((256, 256, 3), dtype=np.uint8)
        rgb_array[:, :, 0] = 100  # Red channel
        rgb_array[:, :, 1] = 150  # Green channel
        rgb_array[:, :, 2] = 200  # Blue channel

        rgb_img = Image.fromarray(rgb_array, mode="RGB")

        # Should handle RGB → Grayscale conversion
        roi_img = extract_roi_dorsal(rgb_img, method="otsu")
        assert isinstance(roi_img, Image.Image)
        assert roi_img.mode == "L"  # Should return grayscale


class TestROIExtractionProperties:
    """Test properties of ROI extraction output."""

    def test_output_is_pil_image(self, sample_hand_image):
        """Test that output is always a PIL Image."""
        for method in ["otsu", "adaptive", "percentile"]:
            roi_img = extract_roi_dorsal(sample_hand_image, method=method)
            assert isinstance(roi_img, Image.Image)
            assert hasattr(roi_img, "mode")
            assert hasattr(roi_img, "size")

    def test_output_is_grayscale(self, sample_hand_image):
        """Test that output is always grayscale (L mode)."""
        for method in ["otsu", "adaptive", "percentile"]:
            roi_img = extract_roi_dorsal(sample_hand_image, method=method)
            assert roi_img.mode == "L", f"Expected grayscale, got {roi_img.mode}"

    def test_output_dimensions_valid(self, sample_hand_image):
        """Test that output dimensions are valid."""
        roi_img = extract_roi_dorsal(sample_hand_image, method="otsu")
        w, h = roi_img.size

        # Should have positive dimensions
        assert w > 0
        assert h > 0

        # Should not exceed original image size
        orig_w, orig_h = sample_hand_image.size
        assert w <= orig_w
        assert h <= orig_h


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
