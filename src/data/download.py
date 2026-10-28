"""
Automated, reproducible dataset downloader for Kaggle / UCI benchmark datasets.
Usage:
    python -m src.data.download [dataset_id]
"""
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
        "type": "uci_nested_zip",
        "inner_zip": "student.zip",
        "inner_file": "student-mat.csv",
        "dest": "data/raw/student_performance.csv"
    },
    "appliances_energy": {
        "url": "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
        "type": "zip",
        "inner_file": "energydata_complete.csv",
        "dest": "data/raw/appliances_energy.csv"
    },
    "kc_house_sales": {
        "url": "https://raw.githubusercontent.com/tanlitung/Datasets/master/kc_house_data.csv",
        "type": "direct_csv",
        "dest": "data/raw/kc_house_sales.csv"
    },
    "life_expectancy": {
        "url": "https://raw.githubusercontent.com/EvidenceN/DataSets/master/Life%20Expectancy%20Data.csv",
        "type": "direct_csv",
        "dest": "data/raw/life_expectancy.csv"
    }
}


def download_dataset(dataset_id: str, force: bool = False) -> Path:
    """Download and extract a registered dataset to data/raw/."""
    if dataset_id not in DOWNLOAD_SOURCES:
        raise ValueError(f"Unknown dataset_id '{dataset_id}'. Available: {list(DOWNLOAD_SOURCES.keys())}")

    root = get_project_root()
    cfg = DOWNLOAD_SOURCES[dataset_id]
    dest_path = root / cfg["dest"]

    if dest_path.exists() and not force:
        print(f"Dataset '{dataset_id}' already present at '{dest_path}'.")
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching '{dataset_id}' from {cfg['url']}...")

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

    print(f"Successfully saved to: {dest_path} ({dest_path.stat().st_size:,} bytes)")
    return dest_path


def download_all(force: bool = False) -> None:
    """Download all registered benchmark datasets."""
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
