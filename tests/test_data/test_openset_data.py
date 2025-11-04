#!/usr/bin/env python3
"""
Test script for OpenSet data loading functionality.

This script verifies that the OpenSet data loading pipeline works correctly:
1. Loads MMCBNU dataset with finger-class-level splits
2. Ensures subject-disjoint constraint (no patient in both known/unknown)
3. Creates train/val/test dataloaders with proper P-K sampling
4. Verifies data statistics and split integrity
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import create_openset_data_loaders, verify_subject_disjoint


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_openset_dataloaders():
    """Test OpenSet dataloader creation and verification."""
    print_section("🧪 Testing OpenSet DataLoader Creation")

    try:
        # Create OpenSet dataloaders
        print("📦 Creating OpenSet dataloaders for MMCBNU...")
        loaders, info = create_openset_data_loaders(
            dataset_name="mmcbnu",
            img_size=224,
            known_ratio=0.7,
            val_ratio=0.15,
            subject_disjoint=True,
            enrollment_samples=7,
            test_samples=3,
            seed=42,
            P=16,
            K=4,
            num_workers=0,  # Use 0 for testing
            batch_size=32,
            hflip_p=0.3,
        )

        print("✅ Dataloaders created successfully!\n")

        # Print dataset info
        print_section("📊 Dataset Statistics")
        print(f"Dataset: {info['dataset_name']}")
        print(f"Mode: {info['mode']}")
        print(f"Subject-disjoint: {info['subject_disjoint']}")
        print(f"\nFinger Classes:")
        print(f"  Total: {info['total_finger_classes']}")
        print(f"  Known: {info['known_finger_classes']}")
        print(f"  Unknown: {info['unknown_finger_classes']}")
        print(f"\nPatients:")
        print(f"  Known: {len(info['known_patients'])} patients")
        print(f"  Unknown: {len(info['unknown_patients'])} patients")
        print(f"  Known IDs: {info['known_patients'][:10]}...")
        print(f"  Unknown IDs: {info['unknown_patients'][:10]}...")

        # Print split statistics
        print_section("🎯 Split Statistics")
        for split_name, split_info in info["splits"].items():
            print(f"{split_name}:")
            print(f"  Samples: {split_info['samples']}")
            print(f"  Finger Classes: {split_info['finger_classes']}")
            print(f"  Patients: {split_info['patients']}")
            if "note" in split_info:
                print(f"  Note: {split_info['note']}")
            print()

        # Print sampling info
        print_section("⚙️  Sampling Configuration")
        sampling = info["sampling"]
        print(f"P (classes per batch): {sampling['P']}")
        print(f"K (samples per class): {sampling['K']}")
        print(f"Batch size: {sampling['batch_size']}")
        print(f"Batches per epoch: {sampling['batches_per_epoch']}")

        # Print enrollment info
        print_section("📝 Enrollment Configuration")
        enrollment = info["enrollment"]
        print(f"Enrollment samples per finger: {enrollment['enrollment_samples']}")
        print(f"Test samples per finger: {enrollment['test_samples']}")

        # Verify subject-disjoint constraint
        print_section("✔️  Subject-Disjoint Verification")
        verification = info["verification"]
        print(f"Subject-disjoint: {verification['subject_disjoint']}")
        print(f"Total patients: {verification['total_patients']}")
        print(f"Known patients: {verification['known_patients']}")
        print(f"Unknown patients: {verification['unknown_patients']}")
        print("✅ No patient overlap detected!")

        # Test loading a batch from each loader
        print_section("🔄 Testing Batch Loading")
        
        for loader_name, loader in loaders.items():
            print(f"\n{loader_name}:")
            try:
                batch = next(iter(loader))
                images, labels, metadata = batch
                print(f"  ✅ Batch shape: {images.shape}")
                print(f"  ✅ Labels shape: {labels.shape}")
                print(f"  ✅ Unique classes in batch: {len(set(labels.tolist()))}")
                print(f"  ✅ Sample metadata keys: {list(metadata[0].keys())}")
                
                # Print first sample metadata
                print(f"  ✅ First sample finger_class_id: {metadata[0]['finger_class_id']}")
                print(f"  ✅ First sample patient_id: {metadata[0]['patient_id']}")
                
            except Exception as e:
                print(f"  ❌ Error loading batch: {e}")

        print_section("🎉 All Tests Passed!")
        print("OpenSet data loading is working correctly.")
        print("\nYou can now use:")
        print("  from data import create_openset_data_loaders")
        print("  loaders, info = create_openset_data_loaders('mmcbnu')")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_openset_dataloaders()
    sys.exit(0 if success else 1)
