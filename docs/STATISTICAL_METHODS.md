# Statistical Methods & Diagnostic Framework

This document outlines the theoretical foundations, mathematical definitions, null hypotheses ($H_0$), decision rules, and practical limitations for all regression evaluation metrics and statistical diagnostic tests implemented in **Regression Learning Lab**.

---

## 1. Regression Performance Metrics

### 1.1 Mean Absolute Error (MAE)
- **Mathematical Formula**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Interpretation**: The average absolute distance between observed and predicted values, expressed in the same units as $y$.
- **Strengths & Limitations**: Robust to extreme outliers; non-differentiable at $0$.

### 1.2 Mean Squared Error (MSE) & Root Mean Squared Error (RMSE)
- **Mathematical Formula**:
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2, \quad \text{RMSE} = \sqrt{\text{MSE}}$$
- **Interpretation**: RMSE measures the standard deviation of residuals. Due to squaring, it heavily penalizes large errors.
- **Strengths & Limitations**: Differentiable and well-behaved mathematically; sensitive to outliers.

### 1.3 Coefficient of Determination ($R^2$)
- **Mathematical Formula**:
  $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
- **Interpretation**: The proportion of variance in the response variable explained by the linear model.
- **Limitation**: Monotonically increases (or stays constant) as additional regressors are added, regardless of statistical relevance.

### 1.4 Adjusted $R^2$ ($R^2_{adj}$)
- **Mathematical Formula**:
  $$R^2_{adj} = 1 - \left( \frac{1 - R^2}{n - p - 1} \right)(n - 1)$$
  where $n$ is sample size and $p$ is the number of predictors.
- **Interpretation**: Penalizes model complexity; only increases if the newly added feature reduces error variance more than expected by chance.

---

## 2. The Five Classical Gauss-Markov Assumptions

To ensure that Ordinary Least Squares (OLS) estimates $\hat{\beta}$ are **BLUE** (Best Linear Unbiased Estimators), the following five conditions must hold:

```text
1. Linearity:           E[y | X] = Xβ
2. Independence:        Cov(ε_i, ε_j) = 0  for all i ≠ j
3. Homoscedasticity:    Var(ε_i | X) = σ²  for all i
4. Normality:           ε | X ~ N(0, σ² I)
5. No Multicollinearity: rank(X) = p + 1 (Full column rank)
```

---

## 3. Formal Assumption Tests & Diagnostic Tools

### 3.1 Assumption 1: Linearity

#### A. Graphical Diagnostic: Residuals vs. Fitted Plot
- **Method**: Scatter plot of unstandardized residuals $e_i$ on the vertical axis against fitted values $\hat{y}_i$ on the horizontal axis, overlaid with a non-parametric LOWESS (Locally Weighted Scatterplot Smoothing) trendline.
- **Ideal Pattern**: Residuals randomly dispersed around the horizontal zero line ($e = 0$) with no curved or U-shaped trends.

#### B. Statistical Test: Ramsey RESET Test
- **Purpose**: Detects general functional form misspecification (omitted non-linearities).
- **Auxiliary Regression**: Regresses $y$ on original $X$ plus powers of fitted values ($\hat{y}^2, \hat{y}^3$):
  $$y = X\beta + \gamma_1 \hat{y}^2 + \gamma_2 \hat{y}^3 + u$$
- **Hypotheses**:
  - $H_0: \gamma_1 = \gamma_2 = 0$ (Linear specification is correct)
  - $H_1: \gamma_1 \neq 0 \text{ or } \gamma_2 \neq 0$ (Model is misspecified)
- **Decision Rule**: Reject $H_0$ if $p < 0.05$. Remediation includes polynomial features, log-transformations, or interaction terms.

---

### 3.2 Assumption 2: Independence of Errors (No Autocorrelation)

#### A. Graphical Diagnostic: Autocorrelation Function (ACF) Plot
- **Method**: Plots sample autocorrelation coefficients $r_k = \frac{\sum e_t e_{t-k}}{\sum e_t^2}$ across lags $k = 1, \dots, 20$ with Bartlett's $95\%$ confidence bounds.

#### B. Statistical Test: Durbin-Watson Statistic ($DW$)
- **Test Statistic**:
  $$DW = \frac{\sum_{i=2}^n (e_i - e_{i-1})^2}{\sum_{i=1}^n e_i^2} \approx 2(1 - \hat{\rho})$$
- **Interpretation**:
  - $DW \approx 2.0$: No first-order serial correlation.
  - $DW < 1.5$: Positive autocorrelation.
  - $DW > 2.5$: Negative autocorrelation.

#### C. Statistical Test: Ljung-Box $Q$-Test
- **Hypotheses**:
  - $H_0$: Error terms up to lag $m$ are uncorrelated.
  - $H_1$: Significant autocorrelation exists at one or more lags.

---

### 3.3 Assumption 3: Homoscedasticity (Constant Error Variance)

#### A. Graphical Diagnostic: Scale-Location Plot
- **Method**: Plots $\sqrt{|r_i^*|}$ (square root of standardized residuals) against fitted values $\hat{y}_i$.
- **Ideal Pattern**: A horizontal smoothing line with evenly scattered points. A fan or funnel shape indicates heteroscedasticity.

#### B. Statistical Test: Breusch-Pagan Test
- **Auxiliary Regression**: Regresses squared scaled residuals on regressors:
  $$\frac{e_i^2}{\hat{\sigma}^2} = \delta_0 + \delta_1 X_{i1} + \dots + \delta_p X_{ip} + v_i$$
- **Hypotheses**:
  - $H_0: \delta_1 = \dots = \delta_p = 0$ (Homoscedasticity)
  - $H_1$: Heteroscedasticity is present
- **Test Statistic**: $LM = \frac{1}{2} SS_{reg} \sim \chi^2(p)$ under $H_0$.

#### C. Remediation
- Use White's Heteroscedasticity-Consistent Covariance Estimator (HC3) or Weighted Least Squares (WLS).

---

### 3.4 Assumption 4: Normality of Residuals

#### A. Graphical Diagnostic: Normal Q-Q Plot
- **Method**: Plots empirical standardized residual quantiles against theoretical standard normal quantiles $\Phi^{-1}(p_i)$.
- **Ideal Pattern**: Points fall closely along the straight diagonal reference line. Divergence at the tails indicates fat tails (kurtosis) or skewness.

#### B. Statistical Test: Jarque-Bera Test
- **Test Statistic**:
  $$JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right) \sim \chi^2(2)$$
  where $S$ is sample skewness and $K$ is sample kurtosis.
- **Hypotheses**:
  - $H_0: S = 0 \text{ and } K = 3$ (Residuals are normally distributed)
  - $H_1$: Residuals significantly depart from normality.

#### C. Statistical Test: Shapiro-Wilk Test
- Particularly powerful for smaller sample sizes ($n < 2000$).

---

### 3.5 Assumption 5: Absence of Multicollinearity

#### A. Diagnostic Metric: Variance Inflation Factor (VIF)
- **Mathematical Formula**:
  $$VIF_j = \frac{1}{1 - R_j^2}$$
  where $R_j^2$ is the $R^2$ obtained from regressing feature $X_j$ on all other $p-1$ predictors.
- **Decision Thresholds**:
  - $VIF = 1$: Completely orthogonal (uncorrelated).
  - $1 < VIF < 5$: Moderate, acceptable collinearity.
  - $VIF \ge 5$: High collinearity; inflation of parameter standard error by $\sqrt{VIF} \ge 2.23$.
  - $VIF \ge 10$: Severe collinearity; matrix inversion instability.

#### B. Diagnostic Metric: Matrix Condition Number ($\kappa$)
- **Definition**: $\kappa(X) = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}$ from singular value decomposition (SVD).
- **Rule of Thumb**: $\kappa > 30$ signifies moderate to severe collinearity.

---

## 4. Outliers, Leverage, and Influential Observations

### 4.1 Hat Matrix & Leverage ($h_{ii}$)
- **Formula**: $H = X(X^T X)^{-1} X^T$; $h_{ii}$ represents the leverage of the $i$-th observation in feature space.
- **Cutoff**: Points with $h_{ii} > \frac{2(p+1)}{n}$ have high leverage.

### 4.2 Cook's Distance ($D_i$)
- **Formula**:
  $$D_i = \frac{\sum_{j=1}^n (\hat{y}_j - \hat{y}_{j(i)})^2}{(p + 1) s^2} = \frac{r_i^2}{p + 1} \left( \frac{h_{ii}}{1 - h_{ii}} \right)$$
- **Interpretation**: Quantifies how much all predicted values change when the $i$-th observation is omitted.
- **Cutoff**: Observations with $D_i > \frac{4}{n}$ (or $D_i > 1$) warrant careful investigation.
