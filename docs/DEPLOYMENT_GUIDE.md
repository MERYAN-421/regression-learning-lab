# Streamlit Community Cloud Deployment Guide

**Project**: Regression Learning Lab  
**Repository**: [https://github.com/MERYAN-421/regression-learning-lab](https://github.com/MERYAN-421/regression-learning-lab)  
**Target Branch**: `main`  
**Entrypoint**: `app.py`  
**Target Python Version**: `3.11`  
**Status**: Deployment Prepared — Live Verification Pending Owner Authorization  

---

## 1. Overview & Architecture Prerequisites

Regression Learning Lab is an interactive educational platform designed for multiple linear regression pedagogy, exploratory data analysis, leakage-isolated feature selection, OLS estimation, assumption diagnostics, Pareto model comparison, and academic reporting.

To run reliably on Streamlit Community Cloud, the repository includes:
1. **Root Entrypoint**: `app.py` initializes the Streamlit application, global configurations, navigation, and multi-user session state.
2. **Deterministic Dependencies**: `requirements.txt` specifies tested Python 3.11 packages with upper bounds preventing breaking updates.
3. **Hardened Configuration**: `.streamlit/config.toml` enforces strict CORS and XSRF protections (`enableCORS = true`, `enableXsrfProtection = true`).
4. **Automated Dataset Provisioning**: `scripts/download_datasets.py` and `src/data/loader.py` securely acquire and cryptographically verify all 4 benchmark datasets via SHA-256 checksums if local raw files are not packaged.
5. **Session Isolation**: All models, diagnostics, candidate selections, and reports are encapsulated strictly within Streamlit's per-session state (`st.session_state`).

---

## 2. Streamlit Community Cloud Deployment Steps (Repository Owner)

Streamlit Community Cloud requires OAuth authorization from the repository owner to provision compute and link webhooks to GitHub. Follow these exact steps:

### Step 1: Sign In to Streamlit Community Cloud
1. Navigate to [https://share.streamlit.io/](https://share.streamlit.io/).
2. Click **Sign in with GitHub** using the account owning `MERYAN-421/regression-learning-lab`.

### Step 2: Create New Application
1. In the Streamlit Cloud dashboard, click the **Create app** button (or **New app**).
2. Select **I already have an app**.
3. Fill in the repository configuration:
   - **Repository**: `MERYAN-421/regression-learning-lab`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL (optional)**: e.g., `regression-learning-lab.streamlit.app`

### Step 3: Configure Advanced Settings (Python 3.11)
1. Click **Advanced settings** at the bottom of the deployment modal.
2. Under **Python version**, select **3.11**. *(Note: Python 3.11 is mandatory to ensure compatibility with statistical packages and type annotations).*
3. **Secrets**: No external API keys, database credentials, or Kaggle tokens are required. Leave the secrets field empty.
4. Click **Save**.

### Step 4: Deploy and Monitor Build Logs
1. Click **Deploy!**.
2. Streamlit Cloud will clone the repository, install packages from `requirements.txt`, execute `app.py`, and bind the public URL.
3. View the terminal logs in the bottom-right drawer. Ensure:
   - Python 3.11 environment is active.
   - All packages install with 0 errors.
   - Streamlit starts without unhandled exceptions.

---

## 3. Dataset Acquisition in Cloud Environment

The application supports four benchmark datasets:
1. **Student Performance (Mathematics)** (~395 rows)
2. **Appliances Energy Prediction** (~19,735 rows)
3. **King County House Sales** (~21,613 rows)
4. **Life Expectancy (WHO)** (~2,938 rows)

Because raw data files (`data/raw/*.csv`) are excluded from Git to comply with repository size guidelines and licensing best practices:
- **Automatic On-Demand Download**: When a user selects a dataset from the **Dataset Library** page, `src/data/loader.py` checks if the file exists locally. If not found, it downloads the verified archive from the permanent public mirrors (e.g., UC Irvine Machine Learning Repository) directly into `data/raw/`.
- **Integrity Verification**: The file is checked against its canonical SHA-256 hash defined in `configs/datasets.yaml`. If the hash does not match, execution halts with a clear error preventing corrupted data execution.
- **Pre-provisioning Option**: Alternatively, the cloud build or startup script can invoke:
  ```bash
  python scripts/download_datasets.py
  ```
  to pre-populate all datasets before user traffic arrives.

---

## 4. Production Security & Secrets Policy

1. **No Hardcoded Credentials**: The repository strictly excludes `.env`, `credentials.json`, `*.pem`, `*.key`, and `.streamlit/secrets.toml`.
2. **CORS & XSRF Protection**: Production `.streamlit/config.toml` configures:
   ```toml
   [server]
   enableCORS = true
   enableXsrfProtection = true
   ```
3. **HTML Sanitization**: User-authored content in Academic Reports is sanitized via `src/reporting/html_renderer.py:escape_text` to prevent Cross-Site Scripting (XSS).
4. **Final Test Lock**: Test set evaluations remain strictly isolated and can only be unlocked through intentional, logged user confirmation. Test partitions are never accessible to automated feature selection or model comparison.

---

## 5. Post-Deployment Verification Checklist

Once the cloud application is live, verify the following gates:
- [ ] Application loads at the public URL without startup errors.
- [ ] All 9 sidebar pages navigate cleanly.
- [ ] **Student Performance** runs through EDA, Pearson Feature Selection (k=5), OLS fitting, Diagnostics, Comparison, and Academic Report generation.
- [ ] Report outputs (Markdown, HTML, JSON manifest, Reproduction ZIP) download successfully.
- [ ] Open a second incognito browser session to verify multi-user session state isolation.
