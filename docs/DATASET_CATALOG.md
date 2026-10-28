# Dataset Catalog Specification

This document establishes the registration criteria, metadata schema, and candidate datasets for **Regression Learning Lab**. Full data downloading, caching, and preprocessing will be executed in **Milestone 1**.

---

## 1. Dataset Inclusion Criteria

To ensure authentic pedagogical value for Multiple Linear Regression, candidate datasets must adhere strictly to the following constraints:

1. **Native Dimensionality Constraint (14–39 Raw Predictive Features)**:
   - **Mandatory Rule**: The dataset must naturally contain between **14 and 39 available predictive features** in its original, raw tabular form.
   - **No Arbitrary Subsetting**: Datasets originating with high dimensionality (e.g., 70–80+ columns) where an arbitrary subset of features is hand-picked are **strictly prohibited**. The complexity and collinearity must reflect the genuine raw data structure.
   - *Rationale*: Fewer than 14 features provide insufficient challenge for feature selection algorithms (Filter, Wrapper, Lasso); more than 40 raw features overwhelm students reading econometric summary tables.
2. **Continuous Response Variable ($y$)**:
   - The target variable must be strictly continuous (ratio or interval scale).
   - Binary classification, multinomial, or discrete ordinal tasks without numeric spacing are excluded.
3. **No Artificial Sample Size Ceilings**:
   - Sample sizes vary naturally across domains (from hundreds to tens of thousands of instances) without imposing arbitrary minimum or maximum row limits.
4. **Authentic Statistical Phenomena**:
   - Datasets should exhibit real-world phenomena such as multicollinear feature clusters, varying feature scales, skewness, or potential leverage outliers.

---

## 2. Dataset Metadata Schema

Each dataset registered in `configs/datasets.yaml` follows this standardized schema:

```yaml
id: string                     # Unique snake_case identifier
name: string                   # Human-readable title
source: string                 # Kaggle / UCI source repository
domain: string                 # Application domain
target_variable: string        # Exact column name of continuous response y
target_type: "continuous"
target_unit: string            # Unit of measurement
raw_feature_count: integer     # Number of native features in raw table
usable_feature_range: string   # Candidate predictive features after excluding raw ID/date
statistical_challenges:        # Real-world regression challenges present
  - string
status: "verified_candidate" | "unverified_excluded"
```

---

## 3. Verified Candidate Datasets (Primary Candidates for Milestone 1)

### Candidate 1: Appliances Energy Prediction
- **ID**: `appliances_energy`
- **Source**: UCI Machine Learning Repository / Kaggle (`loveall/appliances-energy-prediction`)
- **Domain**: Energy Consumption & Smart Building IoT
- **Target Variable**: `Appliances` (Energy use in Wh)
- **Raw Feature Count**: 28 predictive features (excluding target; includes lights, indoor temperatures T1–T9, humidities RH_1–RH_9, outdoor weather variables T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint, and random variables).
- **Usable Predictors**: 26–28 continuous regressors.
- **Pedagogical Value**: Ideal for demonstrating collinearity among indoor sensor clusters, exogenous weather variables, and residual autocorrelation.

### Candidate 2: House Sales in King County, USA
- **ID**: `kc_house_sales`
- **Source**: Kaggle (`harlfoxem/housesalesprediction`)
- **Domain**: Real Estate Valuation & Urban Economics
- **Target Variable**: `price` (Sale price in USD)
- **Raw Feature Count**: 20 candidate features (excluding target; bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15).
- **Usable Predictors**: 18–19 features (excluding non-predictive id and date strings).
- **Pedagogical Value**: Classic example of right-skewed target requiring logarithmic transformation, high physical collinearity between area metrics, and spatial location effects.

### Candidate 3: Life Expectancy (WHO)
- **ID**: `life_expectancy`
- **Source**: Global Health Observatory / Kaggle (`kumarajarshi/life-expectancy-who`)
- **Domain**: Public Health & Socio-Economics
- **Target Variable**: `Life expectancy` (Years)
- **Raw Feature Count**: 21 predictive features (Adult Mortality, infant deaths, Alcohol, percentage expenditure, Hepatitis B, Measles, BMI, under-five deaths, Polio, Total expenditure, Diphtheria, HIV/AIDS, GDP, Population, thinness 1-19 years, thinness 5-9 years, Income composition of resources, Schooling, Status, Country, Year).
- **Usable Predictors**: 21 features.
- **Pedagogical Value**: Explores missing data imputation in developing nations, high correlation between demographic indicators, and non-linear economic development curves.

### Candidate 4: Student Performance
- **ID**: `student_performance`
- **Source**: UCI Machine Learning Repository / Kaggle (`uciml/student-alcohol-consumption`)
- **Domain**: Educational Data Mining
- **Target Variable**: `G3` (Final exam grade, 0–20 scale)
- **Raw Feature Count**: 32 candidate attributes in `student-mat.csv`.
- **Usable Predictors**: 30–32 features (demographic, parental education, study time, alcohol consumption, absences, and previous period grades).
- **Pedagogical Value**: Evaluates mixed tabular feature types (binary, categorical, continuous), zero-inflation at grade boundaries, and feature selection stability across demographic vs. behavioral predictors.

---

## 4. Unverified & Excluded Datasets Audit

| Dataset ID | Original Raw Columns | Evaluation Outcome | Rationale |
| :--- | :--- | :--- | :--- |
| `ames_house_prices` | 81 columns | **Excluded** (`unverified_excluded`) | Raw feature count (>39) significantly exceeds the upper threshold. Hand-picking 36 features violates the natural raw feature criterion. |
| `concrete_compressive_strength` | 9 columns | **Excluded** (`unverified_excluded`) | Raw feature count (8 predictive features) falls below the minimum 14-feature threshold. |

---

## 5. Milestone 1 Implementation Roadmap
During Milestone 1:
1. Candidate CSV schemas and column types will be systematically validated against `configs/datasets.yaml`.
2. Raw data will be cached under `data/raw/`.
3. Standardized train/test partitioning pipelines will be implemented in `src/data/`.
