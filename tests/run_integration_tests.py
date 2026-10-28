"""
Standalone Integration Test Runner for Milestone 1.
Locates or downloads the benchmark datasets, verifies cryptographic signatures,
executes the integration test suite, and prints a comprehensive audit report.

Usage:
    python tests/run_integration_tests.py
"""
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_registry, check_dataset_availability
from src.data.download import verify_dataset_integrity, download_dataset, DOWNLOAD_SOURCES


def run_integration_workflow():
    print("=" * 70)
    print("  Regression Learning Lab - Real-Dataset Integration Audit")
    print("=" * 70)

    reg = get_registry()
    dataset_ids = list(DOWNLOAD_SOURCES.keys())

    print("\n[Step 1/3] Checking dataset files and cryptographic signatures on disk...")
    all_ready = True
    for ds_id in dataset_ids:
        meta = reg.get_dataset(ds_id)
        exists = check_dataset_availability(meta)
        if not exists:
            print(f"  [MISSING] '{ds_id}' at {meta.local_path}")
            all_ready = False
        else:
            is_valid = verify_dataset_integrity(ds_id)
            if is_valid:
                print(f"  [OK] Verified: '{ds_id}' (SHA-256 matches catalog signature)")
            else:
                print(f"  [MISMATCH] Checksum Mismatch for '{ds_id}'. Re-download required.")
                all_ready = False

    if not all_ready:
        print("\n[Step 2/3] Fetching missing/corrupted datasets from verified sources...")
        for ds_id in dataset_ids:
            meta = reg.get_dataset(ds_id)
            if not check_dataset_availability(meta) or not verify_dataset_integrity(ds_id):
                print(f"  Downloading '{ds_id}'...")
                download_dataset(ds_id, force=True)
    else:
        print("\n[Step 2/3] All real datasets present and cryptographically verified.")

    print("\n[Step 3/3] Running integration tests via pytest...")
    test_target = PROJECT_ROOT / "tests" / "test_integration_datasets.py"
    cmd = [sys.executable, "-m", "pytest", "-v", str(test_target)]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("  INTEGRATION TEST AUDIT PASSED: All real datasets verified!")
    else:
        print("  INTEGRATION TEST AUDIT FAILED. Review output above.")
    print("=" * 70)

    sys.exit(result.returncode)


if __name__ == "__main__":
    run_integration_workflow()
