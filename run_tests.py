"""
Main test runner for the OpenSet project.

Run this script to execute all tests.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("🚀 OpenSet Project - Test Suite")
    print("=" * 50)

    # Change to project directory
    os.chdir(project_root)

    # Import and run data tests
    try:
        from tests.test_data.test_all import run_all_tests

        print("\n📊 Testing data module...")
        success = run_all_tests()

        if success:
            print("\n✅ All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)
