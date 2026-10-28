# Exploratory Data Analysis (EDA) Laboratory Methodology

This document details the mathematical foundations, descriptive statistical formulas, diagnostic visual representations, and pedagogical principles implemented in the **Exploratory Data Analysis (EDA) Lab** of **Regression Learning Lab**.

---

## 1. Educational Objectives & Architectural Principles

The EDA Lab is designed as an interactive diagnostic workbench enabling users to explore dataset characteristics prior to feature selection and regression modeling.

### Core Architectural Principles:
1. **Partition Scope Isolation**: Exploratory data analysis defaults strictly to the **Training Data partition** to prevent visual data leakage into validation or test sets. Users can toggle between Training, Validation, Test, or Raw Audit views with explicit scope indicators.
2. **Type-Appropriate Visualizations**: Continuous numeric features and discrete categorical variables are handled through distinct mathematical pipelines. Categorical codes are never plotted as continuous distances.
3. **Association vs. Causation Warning**: High correlation magnitudes or steep regression trendlines illustrate statistical association, not causal mechanisms.
4. **Display-Only Performance Sampling**: Datasets exceeding 3,000 points utilize responsive display sampling for interactive Plotly charts, while computing metrics on the complete active partition. The sample size is always explicitly labeled.

---

## 2. Univariate Distribution Analysis

### 2.1 Continuous Numeric Predictors & Target
For each numeric variable $X = \{x_1, x_2, \dots, x_N\}$, the following statistics are calculated:

* **Sample Mean**:
  $$\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$$
* **Sample Standard Deviation** (unbiased, degrees of freedom $N-1$):
  $$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2}$$
* **Median ($Q_2$) and Interquartile Range ($IQR$)**:
  $$IQR = Q_3 - Q_1$$
  Where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile.
* **Sample Skewness** (Fisher-Pearson coefficient of asymmetry):
  $$g_1 = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^3}{\left(\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^{3/2}}$$
  * $g_1 > 0$: Right-skewed (positive tail), e.g., house sales price or energy consumption.
  * $g_1 < 0$: Left-skewed (negative tail).
  * $g_1 \approx 0$: Symmetric distribution.
* **Sample Kurtosis** (Fisher excess kurtosis, normal distribution = 0):
  $$g_2 = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^4}{\left(\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^2} - 3$$

### 2.2 IQR-Based Outlier Detection
Outliers are flagged descriptively using Tukey's fences:
$$\text{Lower Fence} = Q_1 - 1.5 \times IQR, \quad \text{Upper Fence} = Q_3 + 1.5 \times IQR$$
Observations falling outside $[\text{Lower Fence}, \text{Upper Fence}]$ are counted and presented for auditing.
> **Pedagogical Note**: Outliers are not automatically deleted. In econometric and regression analysis, extreme points often contain legitimate structural information (e.g., luxury estates in King County House Sales).

### 2.3 Categorical Predictors
Categorical variables are characterized by:
* **Cardinality**: Number of distinct categories $K$.
* **Mode Frequency & Percentage**:
  $$\text{Mode \%} = \frac{\text{Count}(\text{Mode})}{N_{\text{valid}}} \times 100\%$$
* **Missing Value Ratio**:
  $$\text{Missing \%} = \frac{N_{\text{missing}}}{N_{\text{total}}} \times 100\%$$
* **Frequency Breakdown Bar Chart**: Displays top categories with exact counts and proportions.

---

## 3. Correlation Analysis: Pearson vs. Spearman

The correlation matrix reveals pairwise linear and monotonic dependencies among numeric predictors.

### 3.1 Pearson Product-Moment Correlation ($r$)
Measures the strength and direction of **linear association** between two variables $X$ and $Y$:
$$r_{xy} = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2}}$$
* Range: $[-1.0, +1.0]$.
* Assumes bivariate linearity and sensitivity to extreme observations.

### 3.2 Spearman Rank Correlation ($\rho$)
Measures the strength and direction of **monotonic association** by computing Pearson's correlation on ranked variables:
$$\rho = 1 - \frac{6 \sum_{i=1}^{N} d_i^2}{N(N^2 - 1)}$$
Where $d_i = \text{rank}(x_i) - \text{rank}(y_i)$.
* Robust against outliers and captures non-linear monotonic relationships (e.g., exponential growth $y = e^x$ yields $\rho = 1.0$ while $r < 1.0$).

### 3.3 Robust Implementation Details
* **Zero-Variance (Constant) Columns**: Excluded automatically from the correlation matrix to prevent division-by-zero errors. Flagged under constant feature warnings.
* **Pairwise Deletion**: Handled via pairwise non-null filtering (`min_periods=10`).
* **Top Pairwise Correlations Table**: Automatically sorts pairs by absolute magnitude $|\text{corr}|$ descending.

---

## 4. Bivariate Feature Relationships

### 4.1 Numeric Predictor vs. Target
* **Scatter Plot**: Shows individual observation points with interactive hover tooltips.
* **Ordinary Least Squares (OLS) Trendline**:
  $$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i, \quad \hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$
* **Diagnostic Statistics**: Displays sample count $N$, Pearson $r$ (with two-tailed $p$-value), Spearman $\rho$, and OLS slope $\hat{\beta}_1$.

### 4.2 Categorical Predictor vs. Target
* **Box Plot**: Plots target distribution across discrete category levels.
* **Summary Table**: Reports group sample count, mean, median, standard deviation, and min/max target values per category.

---

## 5. Target Variable Distribution & Modeling Implications

A dedicated Target Distribution panel educates users on the implications of skewed dependent variables:
* **OLS Sensitivity**: OLS minimizes squared residuals $\sum (y_i - \hat{y}_i)^2$. Extreme values in a heavy-tailed target exert disproportionate leverage on parameter estimates.
* **Evaluation Metric Skew**: When the target is heavily right-skewed (e.g., `Appliances Energy` with skewness $3.39$), RMSE is dominated by large errors in the upper tail.
* **No Pre-Emptive Mutation**: In Milestone 2, the target remains untransformed in its natural units to maintain interpretable error metrics across feature selection benchmarks.
