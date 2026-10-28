"""
Automated, reproducible dataset downloader for Kaggle / UCI benchmark datasets.
Includes provenance tracking and SHA-256 cryptographic checksum verification.

Usage:
    python -m src.data.download [dataset_id]
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from src.utils import get_project_root


DOWNLOAD_SOURCES = {
    "student_performance": {
        "url": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
        "provenance": "UCI Machine Learning Repository (Official Direct Archive, Dataset #320)",
        "canonical_source": "https://www.kaggle.com/datasets/uciml/student-alcohol-consumption",
        "type": "uci_nested_zip",
        "inner_zip": "student.zip",
        "inner_file": "student-mat.csv",
        "dest": "data/raw/student_performance.csv",
        "sha256": "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80",
        "expected_bytes": 56993
    },
    "appliances_energy": {
        "url": "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
        "provenance": "UCI Machine Learning Repository (Official Direct Archive, Dataset #374)",
        "canonical_source": "https://www.kaggle.com/datasets/loveall/appliances-energy-prediction",
        "type": "zip",
        "inner_file": "energydata_complete.csv",
        "dest": "data/raw/appliances_energy.csv",
        "sha256": "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d",
        "expected_bytes": 11979363
    },
    "kc_house_sales": {
        "url": "https://raw.githubusercontent.com/tanlitung/Datasets/master/kc_house_data.csv",
        "provenance": "GitHub Verified Open Mirror (Direct Mirror of Kaggle harlfoxem/housesalesprediction)",
        "canonical_source": "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction",
        "type": "direct_csv",
        "dest": "data/raw/kc_house_sales.csv",
        "sha256": "fb9ce109d108656e64bbff4b7317c076807fd46902609029920defb33be8db29",
        "expected_bytes": 2359557
    },
    "life_expectancy": {
        "url": "https://raw.githubusercontent.com/EvidenceN/DataSets/master/Life%20Expectancy%20Data.csv",
        "provenance": "GitHub Verified Open Mirror (Direct Mirror of Kaggle WHO kumarajarshi/life-expectancy-who)",
        "canonical_source": "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who",
        "type": "direct_csv",
        "dest": "data/raw/life_expectancy.csv",
        "sha256": "872125dd1dd0f9140fbead61df20585a815f5cf47db68f08bf54efaf87963b11",
        "expected_bytes": 333442
    }
}


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_integrity(dataset_id: str) -> bool:
    """Check if the dataset file exists and matches its expected SHA-256 checksum."""
    if dataset_id not in DOWNLOAD_SOURCES:
        return False
    cfg = DOWNLOAD_SOURCES[dataset_id]
    root = get_project_root()
    path = root / cfg["dest"]
    if not path.exists() or not path.is_file():
        return False
    actual_hash = compute_file_sha256(path)
    return actual_hash == cfg["sha256"]


def download_dataset(dataset_id: str, force: bool = False) -> Path:
    """Download, extract, and cryptographically verify a benchmark dataset."""
    if dataset_id not in DOWNLOAD_SOURCES:
        raise ValueError(f"Unknown dataset_id '{dataset_id}'. Available: {list(DOWNLOAD_SOURCES.keys())}")

    root = get_project_root()
    cfg = DOWNLOAD_SOURCES[dataset_id]
    dest_path = root / cfg["dest"]

    if dest_path.exists() and not force:
        # Check integrity
        actual_hash = compute_file_sha256(dest_path)
        if actual_hash == cfg["sha256"]:
            print(f"Dataset '{dataset_id}' present and verified at '{dest_path}'.")
            return dest_path
        else:
            print(f"Checksum mismatch for '{dataset_id}'. Re-downloading...")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching '{dataset_id}' from {cfg['url']}...")
    print(f"Provenance: {cfg['provenance']}")

    req = urllib.request.Request(
        cfg["url"],
        headers={"User-Agent": "RegressionLearningLab/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    dtype = cfg["type"]
    if dtype == "direct_csv":
        with open(dest_path, "wb") as f:
            f.write(content)
    elif dtype == "zip":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with open(dest_path, "wb") as f:
                f.write(z.read(cfg["inner_file"]))
    elif dtype == "uci_nested_zip":
        with zipfile.ZipFile(io.BytesIO(content)) as outer_z:
            inner_bytes = outer_z.read(cfg["inner_zip"])
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_z:
                with open(dest_path, "wb") as f:
                    f.write(inner_z.read(cfg["inner_file"]))

    actual_hash = compute_file_sha256(dest_path)
    if actual_hash != cfg["sha256"]:
        raise ValueError(
            f"Checksum verification failed for '{dataset_id}'!\n"
            f"Expected SHA-256: {cfg['sha256']}\n"
            f"Actual SHA-256:   {actual_hash}"
        )

    print(f"Successfully saved and verified: {dest_path} ({dest_path.stat().st_size:,} bytes, SHA-256: {actual_hash[:16]}...)")
    return dest_path


def download_all(force: bool = False) -> None:
    """Download and verify all registered benchmark datasets."""
    for ds_id in DOWNLOAD_SOURCES:
        try:
            download_dataset(ds_id, force=force)
        except Exception as e:
            print(f"Error downloading '{ds_id}': {e}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and target in DOWNLOAD_SOURCES:
        download_dataset(target, force=True)
    else:
        download_all(force=False)
