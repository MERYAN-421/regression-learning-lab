# Data Storage Directory

This directory is reserved for local benchmark datasets and cached preprocessed files.

## Storage Guidelines
- Raw Kaggle datasets downloaded during Milestone 1 will be placed under `data/raw/`.
- Processed feature matrices and train/test splits will be cached under `data/processed/`.
- Large binary artifacts and uncompressed `.csv` files are ignored by `.gitignore` to keep the Git repository lean.
- Sample lightweight datasets or metadata will be stored in `data/sample/`.
