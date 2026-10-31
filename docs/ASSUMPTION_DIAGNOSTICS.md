# Statistical Assumption Diagnostics in OLS Regression

This document outlines the econometric methodology, formal hypothesis tests, diagnostic plots, and statistical interpretation rules implemented in the **Statistical Assumption Diagnostics Laboratory** (Milestone 4) of **Regression Learning Lab**.

---

## 1. Mathematical Foundations: The Five Gauss-Markov Assumptions

Under the Gauss-Markov theorem, the Ordinary Least Squares (OLS) estimator $\hat{\beta} = (X^T X)^{-1} X^T y$ is the **Best Linear Unbiased Estimator (BLUE)**—meaning it achieves minimum variance among all linear unbiased estimators—if and only if the following five conditions hold:

1. **Linearity in Parameters**: The data-generating process satisfies $y = X\beta + \epsilon$.
2. **Strict Exogeneity**: $E[\epsilon | X] = 0$. Regressors carry zero information about expected error terms.
3. **No Perfect Multicollinearity**: $\text{rank}(X) = p + 1 \le n$. Regressors are linearly independent, ensuring $(X^T X)$ is invertible.
4. **Spherical Errors — Homoscedasticity**: $\text{Var}(\epsilon_i | X) = \sigma^2$. Error variance is constant across all observations.
5. **Spherical Errors — No Autocorrelation**: $\text{Cov}(\epsilon_i, \epsilon_j | X) = 0$ for all $i \ne j$. Error terms are mutually uncorrelated.

---

## 2. Fundamental Methodological Disclaimers

### 2.1 Gauss-Markov vs. Error Normality
- **BLUE Does NOT Require Normality**: The Gauss-Markov theorem requires only the five structural assumptions above. Normality ($\epsilon \sim \mathcal{N}(0, \sigma^2 I)$) is **not** a prerequisite for OLS to be BLUE.
- **Role of Normality**: Normality is introduced to derive exact finite-sample Student-$t$ and Fisher-$F$ distributions for hypothesis tests and confidence intervals.
- **Asymptotic Validity (CLT)**: For moderate to large samples ($n > 50$), the Central Limit Theorem ensures that OLS parameter estimators $\hat{\beta}$ are asymptotically normal, preserving the validity of inference even when residual distributions exhibit moderate skewness or kurtosis.

### 2.2 Exogeneity $E[\epsilon | X] = 0$ Cannot Be Established by Residual Diagnostics
- In-sample OLS residuals $e = y - X\hat{\beta}$ are mathematically constrained to be orthogonal to the design matrix columns ($X^T e = 0$) and sum to zero whenever an intercept is included.
- Consequently, residual tests (RESET, Breusch-Pagan, Jarque-Bera, Durbin-Watson) **cannot test for omitted confounding variables or simultaneous reverse causality**.
- Favorable diagnostic test results must **never** be interpreted as proof of exogeneity or causal validity.

### 2.3 Strict Hypothesis Testing Semantics: Failure to Reject is NOT Proof
- When a test statistic yields $p \ge \alpha$, the correct scientific statement is: **Fail to reject $H_0$** (insufficient evidence against the null hypothesis at significance level $\alpha$).
- The application strictly prohibits declaring that "$H_0$ is proven true" or that "assumptions are verified".

---

## 3. The Five Core Diagnostic Modules

### Module A — Linearity & Functional Form
- **Diagnostic Plot**: **Residuals vs. Fitted Values** with non-parametric **LOWESS smooth curve** ($f = 0.66$) and horizontal zero line ($e = 0$).
- **Formal Hypothesis Test**: **Ramsey RESET Test** (`statsmodels.stats.diagnostic.linear_reset`).
  - $H_0$: Model conditional mean is correctly specified (coefficients on powers $\hat{y}^2, \hat{y}^3$ equal zero).
  - $H_1$: Specification error present (higher-order fitted powers provide significant explanatory power).
  - Test Statistic: $F$-statistic with degrees of freedom $(p_{\text{aux}}, n - k - p_{\text{aux}})$.
- **Consequences**: Functional misspecification can cause biased parameter estimates and inaccurate out-of-sample predictions.
- **Remedies**: Introduce polynomial terms ($x^2$), logarithmic/power transformations, or continuous interaction terms.

### Module B — Independence & Autocorrelation
- **Diagnostic Metric**: **Durbin-Watson Statistic** ($DW = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2}$).
  - Descriptive heuristic: $DW \approx 2.0$ indicates no first-order serial correlation; $DW < 1.4$ indicates positive serial correlation; $DW > 2.6$ indicates negative serial correlation.
  - Note: Durbin-Watson is descriptive and does not produce an automated exact $p$-value.
- **Formal Hypothesis Test**: **Ljung-Box Portmanteau Test** (`statsmodels.stats.diagnostic.acorr_ljungbox`).
  - $H_0$: Residual autocorrelations up to lag $m$ are jointly zero (white noise).
  - $H_1$: At least one autocorrelation up to lag $m$ is non-zero.
  - Default lag selection rule: $m = \min(10, \max(1, \lfloor n / 10 \rfloor))$.
- **Diagnostic Plot**: **Autocorrelation Function (ACF)** with approximate Bartlett 95% confidence bands ($\pm 1.96 / \sqrt{n}$).
- **Consequences**: Serial correlation invalidates conventional OLS standard errors (typically deflating SEs and inflating $t$-statistics under positive autocorrelation), causing spurious significance. Coefficients remain unbiased if exogeneity holds.
- **Remedies**: Autoregressive lag features, seasonal differencing, or Newey-West Heteroscedasticity and Autocorrelation Consistent (HAC) covariance estimators.

### Module C — Homoscedasticity
- **Formal Hypothesis Test**: **Breusch-Pagan Test** (`statsmodels.stats.diagnostic.het_breuschpagan`).
  - $H_0$: Error variance is constant across regressor levels ($\text{Var}(\epsilon_i | X) = \sigma^2$).
  - $H_1$: Error variance depends linearly on regressors ($\sigma_i^2 = h(X_i\gamma)$).
  - Primary decision metric: Lagrange Multiplier (LM) score statistic distributed as $\chi^2(p)$.
- **Advanced Option**: **White's General Heteroscedasticity Test** (`het_white`).
  - Auxiliary regression includes regressors, squares, and cross-products. Guarded against degrees-of-freedom exhaustion when feature count is high.
- **Diagnostic Plot**: **Scale-Location Plot** ($\sqrt{|\text{Standardized Residuals}|}$ vs. $\hat{y}$) with LOWESS trend line.
- **Consequences**: Under heteroscedasticity, OLS is no longer BLUE; conventional standard errors and confidence intervals are mathematically invalid. Point estimates remain unbiased and consistent under exogeneity.
- **Remedies**: Huber-White heteroscedasticity-consistent robust standard errors (HC3), Weighted Least Squares (WLS), or response transformations (e.g. $\log(y)$).

### Module D — Normality of Residuals
- **Formal Hypothesis Test**: **Jarque-Bera Test** (`statsmodels.stats.stattools.jarque_bera`).
  - $H_0$: Error distribution has skewness $S = 0$ and kurtosis $K = 3$ (compatible with Gaussian distribution).
  - $H_1$: Residual distribution is non-normal.
  - Test Statistic: $JB = \frac{n}{6} \left(S^2 + \frac{(K - 3)^2}{4}\right) \sim \chi^2(2)$.
- **Small-Sample Option**: **Shapiro-Wilk Test** (`scipy.stats.shapiro`, $n \le 5,000$).
- **Diagnostic Plots**:
  - **Normal Q-Q Plot**: Ordered standardized residuals against theoretical Gaussian quantiles with reference line.
  - **Residual Histogram**: Frequency distribution overlaid with theoretical Gaussian probability density curve $\mathcal{N}(\mu_e, s_e^2)$.
- **Consequences**: Non-normality affects finite-sample exact $t$ and $F$ tests in small samples ($n < 30$). Large-sample asymptotic inference remains valid by the Central Limit Theorem.

### Module E — Multicollinearity
- **Diagnostic Metric**: **Variance Inflation Factor (VIF)** (`statsmodels.stats.outliers_influence.variance_inflation_factor`).
  - Formula: $\text{VIF}_j = \frac{1}{1 - R_j^2}$, where $R_j^2$ is the coefficient of determination from regressing $X_j$ on all other regressors including an auxiliary constant.
  - The constant/intercept is strictly excluded from VIF reporting.
  - Heuristic thresholds: $\text{VIF} < 5$ (Low / Stable), $5 \le \text{VIF} \le 10$ (Moderate), $\text{VIF} > 10$ (High / Severe).
- **Matrix Condition Number**: $\kappa(X) = \sqrt{\lambda_{\max} / \lambda_{\min}}$. Reports overall numerical sensitivity to data perturbations.
- **Matrix Rank**: Full rank ($\text{rank}(X) = p + 1$) vs. rank-deficient ($\text{rank}(X) < p + 1$).
- **Consequences**: Perfect collinearity makes $(X^T X)$ singular and parameters non-identifiable. Near multicollinearity inflates parameter variances and standard errors, making individual $t$-tests insignificant despite high overall model $R^2$. Multicollinearity does **not** bias coefficient estimates.
- **Remedies**: Drop redundant features, $L_2$ Ridge regularization, or principal component analysis.

---

## 4. Dataset-Specific Diagnostic Applicability Matrix

| Benchmark Dataset | Linearity (RESET) | Serial Autocorrelation (DW, LB, ACF) | Homoscedasticity (BP, Scale-Location) | Normality (JB, Q-Q) | Multicollinearity (VIF, Rank) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Appliances Energy** | Applicable | **Applicable** (Strict temporal split; preserves chronological sequence) | Applicable | Applicable | Applicable |
| **KC House Sales** | Applicable | **Not Applicable** (Cross-sectional; row order has no temporal meaning) | Applicable (Inspects land area & price dispersion) | Applicable | **Critical** (Evaluates near-collinearity: `sqft_living ≈ sqft_above + sqft_basement`) |
| **Life Expectancy** | Applicable | **Not Applicable** (Country panel clusters; concatenating countries into a single sequential series is invalid) | Applicable | Applicable | Applicable (Dummy-coded categorical status) |
| **Student Performance** | Applicable | **Not Applicable** (Cross-sectional; sequential ordering invalid) | Applicable | Applicable (Bounded discrete exam scores) | Applicable (One-hot encoded demographic features) |

---

## 5. Artifact & Experiment Tracking Contract

Diagnostics are encapsulated in strongly-typed dataclasses defined in [`src/diagnostics/result_models.py`](file:///d:/碩一/down_temp/classification/src/diagnostics/result_models.py):

- **`DiagnosticResult`**: Individual diagnostic outcome containing test statistic, $p$-value, $\alpha$, hypothesis statements, decision enum (`reject_null`, `fail_to_reject_null`, `descriptive_only`, `not_applicable`, `insufficient_data`, `numerical_failure`), applicability status, educational interpretations, limitations, recommendations, and SHA-256 reproducibility hash.
- **`DiagnosticSuiteResult`**: Aggregated collection of all diagnostic results executed for a specific model instance and $\alpha$ configuration. Supports JSON export and tabular leaderboard rendering.
