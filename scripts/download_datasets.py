"""
Automated dataset provisioner for Regression Learning Lab.
Used for CI pipelines, fresh container initialization, and cloud deployment setups.
Downloads and verifies SHA-256 cryptographic hashes for all 4 benchmark datasets.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_all, verify_dataset_integrity, DOWNLOAD_SOURCES


def main():
    print("=" * 60)
    print("Regression Learning Lab — Dataset Provisioning Script")
    print("=" * 60)
    print(f"Target datasets: {list(DOWNLOAD_SOURCES.keys())}")
    
    download_all(force=False)
    
    print("\nVerifying integrity...")
    all_ok = True
    for ds_id in DOWNLOAD_SOURCES:
        ok = verify_dataset_integrity(ds_id)
        status = "PASSED (SHA-256 Verified)" if ok else "FAILED"
        print(f"  [{status}] {ds_id}")
        if not ok:
            all_ok = False
            
    if all_ok:
        print("\nAll 4 benchmark datasets provisioned and cryptographically verified.")
        sys.exit(0)
    else:
        print("\nOne or more datasets failed integrity checks.")
        sys.exit(1)


if __name__ == "__main__":
    main()
