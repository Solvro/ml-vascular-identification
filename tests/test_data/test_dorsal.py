"""
Test cases for Dorsal dataset implementation.

Tests scanning, dataset creation, and data loading functionality.
"""
import os
import sys
import unittest
from pathlib import Path

# Add src to path for imports - must be before other local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# ruff: noqa: E402
from src.data.dorsal import DorsalDataset, DorsalScanner, build_dorsal_manifest


class TestDorsalScanner(unittest.TestCase):
    """Test DorsalScanner functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Get absolute path to data directory from project root
        project_root = Path(__file__).parent.parent.parent
        self.data_root = project_root / "data" / "dorsal"

    def test_scan_structure_exists(self):
        """Test if dorsal data directory exists."""
        self.assertTrue(
            self.data_root.exists(),
            f"Dorsal data directory not found: {self.data_root}",
        )

    def test_scan_finds_patients(self):
        """Test scanner finds patient directories."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        df = DorsalScanner.scan(str(self.data_root))

        # Basic structure tests
        self.assertGreater(len(df), 0, "No samples found in dorsal dataset")

        # Check required columns
        required_columns = ["path", "patient_id", "side", "dataset"]
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Check dataset label
        self.assertTrue(
            all(df["dataset"] == "dorsal"), "All samples should be labeled 'dorsal'"
        )

        # Check sides
        valid_sides = {"Left", "Right"}
        self.assertTrue(
            df["side"].isin(valid_sides).all(),
            f"Invalid sides found: {df['side'].unique()}",
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

        df = DorsalScanner.scan(str(self.data_root))

        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        for path in df["path"]:
            ext = Path(path).suffix.lower()
            self.assertIn(ext, valid_extensions, f"Invalid file extension: {ext}")

    def test_patient_consistency(self):
        """Test that patient folders match patient IDs in paths."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        df = DorsalScanner.scan(str(self.data_root))

        for _, row in df.iterrows():
            path = Path(row["path"])
            # Path should be: .../P{pid}/{side}/{filename}
            patient_folder = path.parent.parent.name
            expected_folder = f"P{row['patient_id']}"
            self.assertEqual(
                patient_folder,
                expected_folder,
                f"Patient folder {patient_folder} doesn't match ID {row['patient_id']}",
            )


class TestDorsalDataset(unittest.TestCase):
    """Test DorsalDataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Get absolute path to data directory from project root
        project_root = Path(__file__).parent.parent.parent
        self.data_root = project_root / "data" / "dorsal"

    def test_dataset_creation_auto(self):
        """Test automatic dataset creation without parameters."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            dataset = DorsalDataset()
            self.assertGreater(len(dataset), 0, "Dataset should contain samples")

            # Test dataset properties
            patient_ids = dataset.get_patient_ids()
            self.assertGreater(len(patient_ids), 0, "Should have patient IDs")

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
            dataset = DorsalDataset()

            if len(dataset) > 0:
                img, label, metadata = dataset[0]

                # Check return types and structure
                self.assertIsNotNone(img, "Image should not be None")
                self.assertIsInstance(label, int, "Label should be integer")
                self.assertIsInstance(metadata, dict, "Metadata should be dict")

                # Check metadata contents
                required_keys = {"side", "patient_id", "path"}
                for key in required_keys:
                    self.assertIn(key, metadata, f"Missing metadata key: {key}")

                # Check side value
                self.assertIn(
                    metadata["side"], ["Left", "Right"], "Side should be Left or Right"
                )

        except Exception as e:
            self.fail(f"Dataset getitem failed: {e}")

    def test_patient_samples_retrieval(self):
        """Test getting samples by patient ID."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            dataset = DorsalDataset()
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


class TestDorsalManifest(unittest.TestCase):
    """Test manifest building and caching."""

    def setUp(self):
        """Set up test fixtures."""
        # Get absolute path to data directory from project root
        project_root = Path(__file__).parent.parent.parent
        self.data_root = project_root / "data" / "dorsal"
        self.cache_dir = project_root / "data" / "cache"

    def test_manifest_building(self):
        """Test manifest can be built."""
        if not self.data_root.exists():
            self.skipTest(f"Data directory {self.data_root} not found")

        try:
            df = build_dorsal_manifest()

            self.assertGreater(len(df), 0, "Manifest should contain samples")

            required_columns = ["path", "patient_id", "side", "dataset"]
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
            cache_file = self.cache_dir / "dorsal.parquet"
            if cache_file.exists():
                cache_file.unlink()

            # Build manifest (should create cache)
            df1 = build_dorsal_manifest()
            self.assertTrue(cache_file.exists(), "Cache file should be created")

            # Build again (should load from cache)
            df2 = build_dorsal_manifest()

            # Should be identical
            self.assertEqual(
                len(df1), len(df2), "Cached and fresh manifests should be identical"
            )

        except Exception as e:
            self.fail(f"Manifest caching failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
