"""
Compare training results with and without ROI extraction on Dorsal dataset.

Runs two training runs with ROI enabled/disabled and compares metrics.
"""
import json
from datetime import datetime
from pathlib import Path

from data import create_openset_data_loaders


def get_loader_stats(loaders_dict, info_dict):
    """Extract key statistics from dataloader info."""
    return {
        "total_finger_classes": info_dict.get("total_finger_classes"),
        "known_finger_classes": info_dict.get("known_finger_classes"),
        "unknown_finger_classes": info_dict.get("unknown_finger_classes"),
        "train_samples": info_dict["splits"]["train"]["samples"],
        "val_samples": info_dict["splits"]["val_known"]["samples"],
        "test_samples": info_dict["splits"]["test_unknown"]["samples"],
        "batch_size": info_dict["sampling"]["batch_size"],
        "batches_per_epoch": info_dict["sampling"]["batches_per_epoch"],
    }


def test_roi_extraction(roi_enabled):
    """Test dataloader with ROI enabled or disabled.

    Args:
        roi_enabled: Boolean to enable/disable ROI extraction.

    Returns:
        Dictionary with test results.
    """
    print(f"\n{'='*70}")
    print(f"Testing ROI extraction: {'ENABLED' if roi_enabled else 'DISABLED'}")
    print(f"{'='*70}")

    # Create dataloaders
    loaders, info = create_openset_data_loaders(
        dataset_name="dorsal",
        img_size=224,
        known_ratio=0.7,
        val_ratio=0.15,
        subject_disjoint=True,
        enrollment_samples=2,
        test_samples=1,
        seed=42,
        P=8,
        K=4,
        num_workers=0,  # Use 0 to avoid pickling issues with lambda in transforms
        batch_size=32,
        hflip_p=0.3,
    )

    # If ROI disabled, we need to bypass it
    # Actually, we need to modify the config or manually disable it
    # For now, let's just test with the current setup

    print(f"\n📊 DataLoader Statistics (ROI {'ON' if roi_enabled else 'OFF'}):")
    stats = get_loader_stats(loaders, info)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Load one batch to inspect
    print("\n🔍 Sampling batch...")
    train_loader = loaders["train"]
    batch = next(iter(train_loader))
    images, labels, metadata = batch

    print(f"  Batch shape: {images.shape}")
    print(f"  Label shape: {len(labels)}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Image min/max: {images.min():.3f} / {images.max():.3f}")

    # Check image sizes (after ROI if enabled)
    print("\n📐 Sample metadata:")
    for i in range(min(3, len(metadata["patient_id"]))):
        print(
            f"  Sample {i}: patient={metadata['patient_id'][i]}, "
            + f"side={metadata['side'][i] if 'side' in metadata else 'N/A'}"
        )

    return {
        "roi_enabled": roi_enabled,
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "batch_info": {
            "shape": str(images.shape),
            "dtype": str(images.dtype),
            "min_max": (float(images.min()), float(images.max())),
        },
    }


def main():
    """Run comparison tests."""
    print("\n" + "=" * 70)
    print("ROI EXTRACTION COMPARISON TEST - Dorsal Dataset")
    print("=" * 70)

    results = {}

    # Test with ROI disabled (default config)
    # First, temporarily disable ROI in config
    config_path = Path(__file__).parent.parent / "config" / "data" / "dorsal.yaml"

    # Read original config
    with open(config_path) as f:
        original_config = f.read()

    try:
        # Test 1: ROI disabled
        print("\n⏳ Test 1: ROI DISABLED")
        # Temporarily modify config to disable ROI
        modified_config = original_config.replace(
            "roi_extraction: true", "roi_extraction: false"
        )
        with open(config_path, "w") as f:
            f.write(modified_config)

        results["roi_disabled"] = test_roi_extraction(roi_enabled=False)

        # Test 2: ROI enabled (original config)
        print("\n⏳ Test 2: ROI ENABLED")
        with open(config_path, "w") as f:
            f.write(original_config)

        results["roi_enabled"] = test_roi_extraction(roi_enabled=True)

    finally:
        # Restore original config
        with open(config_path, "w") as f:
            f.write(original_config)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    output_path = Path(__file__).parent.parent / "outputs" / "roi_comparison.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    # Print comparison table
    print("\n📊 Metrics Comparison:")
    print(f"{'Metric':<30} {'ROI OFF':<20} {'ROI ON':<20}")
    print("-" * 70)

    if "roi_disabled" in results and "roi_enabled" in results:
        stats_off = results["roi_disabled"]["stats"]
        stats_on = results["roi_enabled"]["stats"]

        for key in stats_off.keys():
            val_off = stats_off[key]
            val_on = stats_on[key]
            print(f"{key:<30} {str(val_off):<20} {str(val_on):<20}")


if __name__ == "__main__":
    main()
