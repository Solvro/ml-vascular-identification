"""
Test cases for MMCBNU dataset implementation.

Tests scanning, dataset creation, and data loading functionality.
"""
import os
import sys
import unittest
from pathlib import Path

# Add src to path for imports - must be before other local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# ruff: noqa: E402
from src.data.mmcbnu import MMCBNUDataset, MMCBNUScanner, build_mmcbnu_manifest


class TestMMCBNUScanner(unittest.TestCase):
    """Test MMCBNUScanner functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Get absolute path to data directory from project root
        project_root = Path(__file__).parent.parent.parent
        self.data_root = project_root / "data" / "mmcbnu"

    def test_scan_structure_exists(self):
        """Test if MMCBNU data directory exists."""
        self.assertTrue(
            self.data_root.exists(),
            f"MMCBNU data directory not found: {self.data_root}",
        )

    def test_scan_finds_patients(self):
        """Test scanner finds patient directories."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        df = MMCBNUScanner.scan(str(self.data_root))

        # Basic structure tests
        self.assertGreater(len(df), 0, "No samples found in MMCBNU dataset")

        # Check required columns
        required_columns = ["path", "patient_id", "finger", "dataset"]
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Check dataset label
        self.assertTrue(
            all(df["dataset"] == "mmcbnu"), "All samples should be labeled 'mmcbnu'"
        )

        # Check patient ID format
        self.assertTrue(
            df["patient_id"].str.match(r"^\d{3}$").all(),
            "Patient IDs should be 3-digit strings",
        )

    def test_scan_file_extensions(self):
        """Test scanner only includes valid image files."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        df = MMCBNUScanner.scan(str(self.data_root))

        valid_extensions = {".bmp", ".png", ".jpg", ".jpeg"}
        for path in df["path"]:
            ext = Path(path).suffix.lower()
            self.assertIn(ext, valid_extensions, f"Invalid file extension: {ext}")

    def test_rois_directory_handling(self):
        """Test that scanner properly handles ROIs subdirectory."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        # Test with ROIs directory if it exists
        rois_dir = self.data_root / "ROIs"
        if rois_dir.exists():
            df = MMCBNUScanner.scan(str(self.data_root))
            self.assertGreater(len(df), 0, "Should find samples in ROIs directory")

            # Check that paths include ROIs
            sample_path = Path(df.iloc[0]["path"])
            self.assertIn(
                "ROIs", sample_path.parts, "Paths should include ROIs directory"
            )

    def test_finger_types(self):
        """Test that scanner finds expected finger types."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        df = MMCBNUScanner.scan(str(self.data_root))
        finger_types = set(df["finger"].unique())

        # Should contain finger names (exact names depend on dataset structure)
        self.assertGreater(len(finger_types), 0, "Should find finger types")

        # Print found finger types for debugging
        print(f"Found finger types: {sorted(finger_types)}")


class TestMMCBNUDataset(unittest.TestCase):
    """Test MMCBNUDataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Get absolute path to data directory from project root
        project_root = Path(__file__).parent.parent.parent
        self.data_root = project_root / "data" / "mmcbnu"

    def test_dataset_creation_auto(self):
        """Test automatic dataset creation without parameters."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            dataset = MMCBNUDataset()
            self.assertGreater(len(dataset), 0, "Dataset should contain samples")

            # Test dataset properties
            patient_ids = dataset.get_patient_ids()
            self.assertGreater(len(patient_ids), 0, "Should have patient IDs")

            # Test finger types
            fingers = dataset.get_fingers()
            self.assertGreater(len(fingers), 0, "Should have finger types")

            # Test label encoder
            self.assertIsInstance(
                dataset.le, dict, "Label encoder should be a dictionary"
            )

        except Exception as e:
            self.fail(f"Auto dataset creation failed: {e}")

    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            dataset = MMCBNUDataset()

            if len(dataset) > 0:
                img, label, metadata = dataset[0]

                # Check return types and structure
                self.assertIsNotNone(img, "Image should not be None")
                self.assertIsInstance(label, int, "Label should be integer")
                self.assertIsInstance(metadata, dict, "Metadata should be dict")

                # Check metadata contents
                required_keys = {"finger", "patient_id", "path"}
                for key in required_keys:
                    self.assertIn(key, metadata, f"Missing metadata key: {key}")

        except Exception as e:
            self.fail(f"Dataset getitem failed: {e}")

    def test_patient_samples_retrieval(self):
        """Test getting samples by patient ID."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            dataset = MMCBNUDataset()
            patient_ids = dataset.get_patient_ids()

            if len(patient_ids) > 0:
                first_patient = patient_ids[0]
                patient_samples = dataset.get_samples_by_patient(first_patient)

                self.assertGreater(
                    len(patient_samples), 0, "Patient should have samples"
                )
                self.assertTrue(
                    all(patient_samples["patient_id"] == first_patient),
                    "All samples should belong to the same patient",
                )

        except Exception as e:
            self.fail(f"Patient samples retrieval failed: {e}")

    def test_finger_samples_retrieval(self):
        """Test getting samples by finger type."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            dataset = MMCBNUDataset()
            fingers = dataset.get_fingers()

            if len(fingers) > 0:
                first_finger = fingers[0]
                finger_samples = dataset.get_samples_by_finger(first_finger)

                self.assertGreater(len(finger_samples), 0, "Finger should have samples")
                self.assertTrue(
                    all(finger_samples["finger"] == first_finger),
                    "All samples should belong to the same finger type",
                )

        except Exception as e:
            self.fail(f"Finger samples retrieval failed: {e}")


class TestMMCBNUManifest(unittest.TestCase):
    """Test manifest building and caching."""

    def setUp(self):
        """Set up test fixtures."""
        # Get absolute path to data directory from project root
        project_root = Path(__file__).parent.parent.parent
        self.data_root = project_root / "data" / "mmcbnu"
        self.cache_dir = project_root / "data" / "cache"

    def test_manifest_building(self):
        """Test manifest can be built."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            df = build_mmcbnu_manifest()

            self.assertGreater(len(df), 0, "Manifest should contain samples")

            required_columns = ["path", "patient_id", "finger", "dataset"]
            for col in required_columns:
                self.assertIn(col, df.columns, f"Missing column: {col}")

        except Exception as e:
            self.fail(f"Manifest building failed: {e}")

    def test_manifest_caching(self):
        """Test that manifest gets cached properly."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            # Clear any existing cache
            cache_file = self.cache_dir / "mmcbnu.parquet"
            if cache_file.exists():
                cache_file.unlink()

            # Build manifest (should create cache)
            df1 = build_mmcbnu_manifest()
            self.assertTrue(cache_file.exists(), "Cache file should be created")

            # Build again (should load from cache)
            df2 = build_mmcbnu_manifest()

            # Should be identical
            self.assertEqual(
                len(df1), len(df2), "Cached and fresh manifests should be identical"
            )

        except Exception as e:
            self.fail(f"Manifest caching failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
