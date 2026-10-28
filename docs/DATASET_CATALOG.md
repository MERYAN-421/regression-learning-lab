# Dataset Catalog Specification

This document establishes the registration criteria, metadata schema, and pre-cataloged datasets for **Regression Learning Lab**. Actual data downloading, caching, and preprocessing will be executed in **Milestone 1**.

---

## 1. Dataset Inclusion Criteria
To ensure optimal pedagogical value for Multiple Linear Regression, candidate datasets must adhere to the following constraints:

1. **Dimensionality Constraint**:
   - Number of original predictive features ($p$) must fall strictly within **14 to 39 features**.
   - *Rationale*: Datasets with fewer than 14 features lack sufficient complexity for meaningful feature selection trade-offs; datasets with more than 40 features induce excessive cognitive load for learners interpreting OLS summary tables.
2. **Response Variable Constraint**:
   - The target variable ($y$) must be **strictly continuous** (ratio or interval scale).
   - Binary, multinomial, or ordinal targets are excluded to maintain focus on linear regression.
3. **Sample Size Constraint**:
   - Observation count ($n$) must range between **500 and 5,000 instances**.
   - *Rationale*: Sufficient statistical power for hypothesis testing ($n \gg p$) while enabling instantaneous interactive computation in Streamlit.
4. **Data Integrity & Realism**:
   - Must originate from authentic real-world domains (e.g. Kaggle open datasets) and exhibit natural empirical challenges (moderate multicollinearity, slight skewness, or outlier leverage).

---

## 2. Dataset Metadata Schema
Each dataset registered in `configs/datasets.yaml` must define the following schema attributes:

```yaml
id: string               # Unique snake_case identifier (e.g. house_prices)
name: string             # Display title of the dataset
source: string           # Kaggle or authoritative source URI
domain: string           # Practical field of application (e.g. Real Estate, Health Economics)
target_variable: string  # Exact column name of response variable y
target_type: string      # 'continuous'
target_unit: string      # Measurement unit (e.g. USD, MPa, L/100km)
expected_rows: integer   # Approximate sample row count
feature_count: integer   # Number of candidate predictive features (14-39)
statistical_challenges:  # Notable regression phenomena embedded in the dataset
  - string
status: string           # Cataloging status
```

---

## 3. Pre-Cataloged Candidate Datasets (Scheduled for Milestone 1)

### Dataset A: House Prices (Ames Housing Subset)
- **ID**: `house_prices`
- **Domain**: Real Estate Economics
- **Feature Count**: 36 features (LotArea, OverallQual, YearBuilt, TotalBsmtSF, GrLivArea, FullBath, GarageCars, etc.)
- **Expected Sample Size**: 1,460 rows
- **Target Variable**: `SalePrice` (USD)
- **Pedagogical Value**: Perfect for demonstrating right-skewness transformations ($\log(y)$), handling multicollinear spatial variables, and showing how structural features interact.

### Dataset B: Concrete Compressive Strength
- **ID**: `concrete_strength`
- **Domain**: Materials Science & Civil Engineering
- **Feature Count**: 15 features (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age, etc.)
- **Expected Sample Size**: 1,030 rows
- **Target Variable**: `CompressiveStrength` (MPa)
- **Pedagogical Value**: Highlights non-linear component ratios, ingredient trade-offs, and the utility of polynomial feature expansion.

### Dataset C: Vehicle Fuel Efficiency & Emissions
- **ID**: `vehicle_fuel_efficiency`
- **Domain**: Automotive Engineering & Environmental Science
- **Feature Count**: 18 features (Displacement, Cylinders, Gears, CurbWeight, Horsepower, AxleRatio, AerodynamicDrag, etc.)
- **Expected Sample Size**: 3,980 rows
- **Target Variable**: `Comb_L100km` (Liters per 100 km)
- **Pedagogical Value**: Demonstrates strong physical multicollinearity (Displacement vs. Cylinders vs. Weight), illustrating the necessity of VIF screening.

### Dataset D: Medical Cost Personal Datasets
- **ID**: `medical_cost_personal`
- **Domain**: Healthcare Actuarial Science
- **Feature Count**: 14 features (Age, Sex, BMI, Children, Smoker, Region, BloodPressure, ChronicConditions, etc.)
- **Expected Sample Size**: 2,760 rows
- **Target Variable**: `charges` (Annual medical expenditures in USD)
- **Pedagogical Value**: Demonstrates extreme heteroscedasticity across sub-populations and the vital role of categorical interaction terms (`smoker * bmi`).

---

## 4. Pipeline Execution Plan (Milestone 1)
In Milestone 1, the dataset engine will:
1. Verify presence of cached CSV files in `data/raw/`.
2. Parse column types into numeric vs. categorical encodings.
3. Apply stratified quantile splitting to generate reproducible Train ($80\%$) and Test ($20\%$) splits.
4. Expose the preprocessed matrix via `src/data/` for downstream modeling.
