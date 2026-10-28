# Dataset Catalog Specification & Empirical Verification

This document records the official dataset registration specifications, verified raw CSV schemas, empirical audit results, and operational requirements for all benchmark datasets in **Regression Learning Lab**.

---

## 1. Mandatory Inclusion Constraint

Every registered dataset intended for regression model training must naturally satisfy:
$$\mathbf{14 \le \text{Usable Original Predictors} \le 39}$$

Where:
$$\text{Usable Original Predictors} = \text{Total Raw Columns} - |\text{Excluded Columns}| - 1 (\text{Target})$$

- Hand-picking a subset of features from high-dimensional datasets (>39 features) is strictly prohibited.
- One-Hot Encoded dimension expansions do **not** count toward the 14–39 natural raw feature requirement.
- Target variables, non-predictive row identifiers, and dedicated grouping variables are excluded from the model predictor matrix.

---

## 2. Empirically Verified Benchmark Datasets (Milestone 1)

All four datasets have been downloaded, inspected, parsed, and verified against their declared metadata:

| Dataset ID | Domain | Raw Rows | Raw Cols | Excluded Cols | Target Variable | Usable Predictors | Recommended Split | Local File Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`appliances_energy`** | Energy IoT | 19,735 | 29 | `date` (1) | `Appliances` (Wh) | **27 features** | **Temporal** | `data/raw/appliances_energy.csv` |
| **`kc_house_sales`** | Real Estate | 21,613 | 21 | `id`, `date` (2) | `price` (USD) | **18 features** | **Random** | `data/raw/kc_house_sales.csv` |
| **`life_expectancy`** | Public Health | 2,938 | 22 | `Country` (1) | `Life expectancy` (Yrs) | **20 features** | **Group (Country)** | `data/raw/life_expectancy.csv` |
| **`student_performance`** | Education | 395 | 33 | None (0) | `G3` (0-20 scale) | **32 features** | **Random** | `data/raw/student_performance.csv` |

---

## 3. Detailed Dataset Profiles & Empirical Audits

### 3.1 Dataset A: Appliances Energy Prediction
- **ID**: `appliances_energy`
- **Source**: UCI Machine Learning Repository / Kaggle (`loveall/appliances-energy-prediction`)
- **Citation**: Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Energy and Buildings*, 140, 81-97.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Delimiters & Encoding**: Comma `,`, UTF-8
- **Date Column**: `date` (format: `YYYY-MM-DD HH:MM:SS`, frequency: 10 minutes)
- **Target Variable**: `Appliances` (Continuous, energy consumption in Wh)
- **Verified Column List (29 columns)**:
  - Excluded Metadata: `date`
  - Target: `Appliances`
  - 27 Usable Predictors: `lights`, `T1`, `RH_1`, `T2`, `RH_2`, `T3`, `RH_3`, `T4`, `RH_4`, `T5`, `RH_5`, `T6`, `RH_6`, `T7`, `RH_7`, `T8`, `RH_8`, `T9`, `RH_9`, `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`, `rv1`, `rv2`.
- **Verified Split**: **Temporal Split** (earliest 70% Train: 13,814 rows; next 15% Val: 2,960 rows; latest 15% Test: 2,961 rows).
- **Known Empirical Features**: High multicollinear correlation among indoor room temperatures (T1–T9) and humidities (RH1–RH9); diurnal cycles.

---

### 3.2 Dataset B: King County House Sales
- **ID**: `kc_house_sales`
- **Source**: Kaggle (`harlfoxem/housesalesprediction`) / King County Department of Assessments
- **License**: Open Database / Public Domain (CC0 equivalent)
- **Delimiters & Encoding**: Comma `,`, UTF-8
- **Target Variable**: `price` (Continuous, sale price in USD)
- **Verified Column List (21 columns)**:
  - Excluded Metadata: `id` (observation identifier), `date` (transaction date string)
  - Target: `price`
  - 18 Usable Predictors: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`.
- **Verified Split**: **Random Split** (seed 42, Train: 15,129 rows, Val: 3,241 rows, Test: 3,243 rows).
- **Known Empirical Features**: Right-skewed price distribution; severe collinearity between structural living area metrics (`sqft_living = sqft_above + sqft_basement`).

---

### 3.3 Dataset C: Life Expectancy (WHO)
- **ID**: `life_expectancy`
- **Source**: World Health Organization / Kaggle (`kumarajarshi/life-expectancy-who`)
- **License**: WHO Open Data Repository / Public Domain
- **Delimiters & Encoding**: Comma `,`, UTF-8
- **Grouping Variable**: `Country` (193 unique sovereign states)
- **Target Variable**: `Life expectancy` (Continuous, average lifespan in years)
- **Parsing Requirement**: The raw CSV contains padded whitespaces in several column names (e.g., `'Life expectancy '`, `' BMI '`, `'under-five deaths '`, `' HIV/AIDS'`). The Data Engine strips whitespace automatically during loading while maintaining schema consistency.
- **Target Missingness Audit**: The raw CSV contains **10 missing values** in the target column `Life expectancy`. These 10 rows are explicitly excluded before partitioning, resulting in **2,928 eligible observations**.
- **Verified Column List (22 columns)**:
  - Excluded Metadata: `Country` (used as group key; excluded from model predictors to prevent unseen category complications during cross-country evaluation)
  - Target: `Life expectancy`
  - 20 Usable Predictors: `Year`, `Status`, `Adult Mortality`, `infant deaths`, `Alcohol`, `percentage expenditure`, `Hepatitis B`, `Measles`, `BMI`, `under-five deaths`, `Polio`, `Total expenditure`, `Diphtheria`, `HIV/AIDS`, `GDP`, `Population`, `thinness 1-19 years`, `thinness 5-9 years`, `Income composition of resources`, `Schooling`.
- **Verified Split**: **Group Split by Country** (Train: 2,048 rows / ~135 countries, Val: 432 rows / ~29 countries, Test: 448 rows / ~29 countries). Strictly disjoint sets of nations.

---

### 3.4 Dataset D: Student Performance (Mathematics)
- **ID**: `student_performance`
- **Source**: UCI Machine Learning Repository / Kaggle (`uciml/student-alcohol-consumption`)
- **Citation**: Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Delimiters & Encoding**: Semicolon `;`, UTF-8
- **Target Variable**: `G3` (Final academic period grade, integer scale 0–20)
- **Target Nature Note**: While modeled as a continuous response for educational regression benchmarks, $G3$ is technically a bounded integer grade with zero-inflation.
- **Verified Column List (33 columns)**:
  - Excluded Metadata: None (0)
  - Target: `G3`
  - 32 Usable Predictors:
    - 17 Categorical: `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.
    - 15 Numeric: `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`, `G1`, `G2`.
- **Verified Split**: **Random Split** (seed 42, Train: 276 rows, Val: 59 rows, Test: 60 rows).
- **Temporal/Leakage Note on G1 and G2**: Grades $G1$ and $G2$ represent first and second-period scores. In future milestones, students can examine the effect of omitting or retaining intermediate exam grades.

---

## 4. Excluded / Unverified Datasets Audit

| Dataset ID | Raw Columns | Status | Rejection Rationale |
| :--- | :--- | :--- | :--- |
| `ames_house_prices` | 81 columns | `unverified_excluded` | Violates the natural upper bound ($p \le 39$). Arbitrary column subsetting is prohibited. |
| `concrete_compressive_strength` | 9 columns | `unverified_excluded` | Contains only 8 candidate predictors, violating the lower bound ($p \ge 14$). |
