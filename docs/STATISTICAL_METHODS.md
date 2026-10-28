# Statistical Methods & Diagnostic Framework

This document outlines the theoretical foundations, mathematical definitions, null hypotheses ($H_0$), decision rules, and practical limitations for all regression evaluation metrics and statistical diagnostic tests implemented in **Regression Learning Lab**.

---

## 1. Distinction: Gauss–Markov Assumptions vs. Diagnostic Categories

A critical pedagogical distinction in econometrics and regression analysis is the difference between the **Gauss–Markov Assumptions** (required for OLS to be BLUE) and the **Five Diagnostic Categories** evaluated in this learning platform.

### 1.1 The Classical Gauss–Markov Assumptions (Conditions for BLUE)
The Gauss–Markov Theorem states that under the following assumptions, the Ordinary Least Squares (OLS) estimator $\hat{\beta} = (X^T X)^{-1} X^T y$ is the **Best Linear Unbiased Estimator (BLUE)** (i.e., possesses the minimum variance among all linear unbiased estimators):

1. **Linearity in Parameters**: The data-generating process satisfies $y = X\beta + \epsilon$.
2. **Strict Exogeneity / Zero Conditional Mean**: $E[\epsilon | X] = 0$.
   - **Crucial Role**: The regressors $X$ carry no information regarding the expected error. This implies $Cov(X_j, \epsilon) = 0$.
   - **Violation Consequences**: If exogeneity fails (due to omitted variable bias, measurement errors in $X$, or simultaneity/reverse causality), OLS estimators are **biased and inconsistent** ($\hat{\beta} \not\to \beta$). Inconsistency persists regardless of sample size $n$.
3. **No Perfect Multicollinearity**: The design matrix $X$ has full column rank ($\text{rank}(X) = p + 1$). The matrix $(X^TX)$ is non-singular and invertible.
4. **Spherical Errors (Homoscedasticity & Uncorrelated Errors)**:
   - **Homoscedasticity**: $Var(\epsilon_i | X) = \sigma^2$ (constant error variance across all observations).
   - **No Autocorrelation**: $Cov(\epsilon_i, \epsilon_j | X) = 0$ for all $i \neq j$.

> [!IMPORTANT]
> **Normality is NOT a Gauss–Markov Assumption!**  
> OLS estimators are mathematically BLUE regardless of whether the disturbance $\epsilon$ follows a normal distribution.  
> **Why do we test normality?** Normality ($\epsilon \sim \mathcal{N}(0, \sigma^2 I)$) is required for **exact finite-sample statistical inference**—namely, the validity of exact Student's $t$-tests for individual coefficients, omnibus $F$-tests for overall fit, and symmetric confidence intervals. In large samples, the Central Limit Theorem ensures asymptotic normality of $\hat{\beta}$, but in finite samples, heavy-tailed or skewed errors distort p-values and inference.

### 1.2 The Five Diagnostic Categories of the Learning Platform
The lab structures regression analysis into five comprehensive pedagogical inspection categories:
1. **Linearity**: Evaluates functional form correctness.
2. **Independence**: Evaluates lack of serial autocorrelation.
3. **Homoscedasticity**: Evaluates constant variance across predicted levels.
4. **Normality of Residuals**: Evaluates finite-sample inference validity.
5. **Multicollinearity**: Evaluates numerical stability and variance inflation.

---

## 2. Regression Performance Metrics & Error Distinction

### 2.1 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Interpretation**: The average absolute magnitude of errors, in the original units of $y$. Robust to isolated extreme outliers.

### 2.2 Distinction: Root Mean Squared Error (RMSE) vs. Residual Standard Deviation ($s$)
A common point of confusion is conflating sample RMSE with the unbiased residual standard deviation:

- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} = \sqrt{\frac{SS_{res}}{n}}$$
  *Measures the sample standard error of prediction. Because it divides by $n$ rather than degrees of freedom, RMSE is a slightly biased (downward) estimate of the population error standard deviation $\sigma$ in finite samples.*

- **Residual Standard Deviation ($s$ or Standard Error of the Regression $\hat{\sigma}$)**:
  $$s = \sqrt{\frac{SS_{res}}{n - p - 1}} = \sqrt{\frac{\sum_{i=1}^n e_i^2}{n - p - 1}}$$
  *Adjusts for the $p + 1$ degrees of freedom consumed by estimating the regression intercept and $p$ slopes. $s^2$ is an **unbiased estimator** of the true error variance $\sigma^2$ under the Gauss–Markov assumptions.*
  - When $n \gg p$, $\text{RMSE} \approx s$. However, when degrees of freedom are limited, $s$ is strictly larger than $\text{RMSE}$.

### 2.3 Coefficient of Determination ($R^2$) and Adjusted $R^2$ ($R^2_{adj}$)
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

$$R^2_{adj} = 1 - \left( \frac{1 - R^2}{n - p - 1} \right)(n - 1)$$
- $R^2$ indicates the proportion of total response variance explained by the linear model.
- $R^2_{adj}$ penalizes complexity; adding an uninformative predictor will lower $R^2_{adj}$ even though $R^2$ never decreases.

---

## 3. Formal Statistical Diagnostic Tests & Decision Rules

> [!CAUTION]
> **Interpretation of High p-values ($p > \alpha$):**  
> In all statistical hypothesis testing (e.g. Shapiro–Wilk, Breusch–Pagan, Ramsey RESET), obtaining $p > 0.05$ means **"fail to reject $H_0$"**.  
> **Failing to reject $H_0$ is NOT proof that the null hypothesis is true** (*"Absence of evidence is not evidence of absence"*). It simply indicates that the current sample size, signal-to-noise ratio, or test power was insufficient to detect a departure from the null. Students must always corroborate test statistics with visual inspection of diagnostic plots.

---

### 3.1 Category 1: Linearity

#### A. Graphical Diagnostic: Residuals vs. Fitted Plot
- **Method**: Plot sample residuals $e_i = y_i - \hat{y}_i$ against fitted values $\hat{y}_i$, accompanied by a non-parametric LOWESS curve.
- **Interpretation**: A well-specified linear model produces a horizontal band centered around zero with no discernible curvature, funneling, or systematic trend.

#### B. Statistical Test: Ramsey RESET Test (Regression Equation Specification Error Test)
- **Purpose**: Detects omitted non-linearities and general functional misspecification.
- **Auxiliary Model**: $y = X\beta + \gamma_1 \hat{y}^2 + \gamma_2 \hat{y}^3 + u$.
- **Hypotheses**:
  - $H_0: \gamma_1 = \gamma_2 = 0$ (Linear specification is adequate)
  - $H_1: \gamma_1 \neq 0 \text{ or } \gamma_2 \neq 0$ (Functional form misspecified)
- **Decision Rule**: Reject $H_0$ if $p < 0.05$. If rejected, introduce polynomials, log transformations, or interaction terms. If $p \ge 0.05$, there is no significant evidence of polynomial misspecification.

---

### 3.2 Category 2: Independence of Errors (No Serial Autocorrelation)

#### A. Graphical Diagnostic: Autocorrelation Function (ACF) Plot
- **Method**: Plots empirical residual correlation across lags $k = 1, \dots, 20$ with $95\%$ confidence bounds ($\pm \frac{1.96}{\sqrt{n}}$).

#### B. Statistical Test: Durbin–Watson Test ($DW$)
$$DW = \frac{\sum_{i=2}^n (e_i - e_{i-1})^2}{\sum_{i=1}^n e_i^2} \approx 2(1 - \hat{\rho})$$
- **Hypotheses**:
  - $H_0: \rho = 0$ (No first-order serial correlation)
  - $H_1: \rho \neq 0$ (Autocorrelation present)
- **Interpretation**:
  - $DW \approx 2.0$: Evidence consistent with uncorrelated errors.
  - $DW < 1.5$: Positive autocorrelation (standard errors underestimated, $t$-statistics inflated).
  - $DW > 2.5$: Negative autocorrelation.

#### C. Statistical Test: Ljung–Box Test
- **Purpose**: Joint test of multiple autocorrelation lags simultaneously.

---

### 3.3 Category 3: Homoscedasticity (Equal Error Variance)

#### A. Graphical Diagnostic: Scale-Location Plot
- **Method**: Plots $\sqrt{|r_i^*|}$ (square root of standardized residuals) against fitted values $\hat{y}_i$.
- **Interpretation**: Upward or downward sloping curves indicate variance changing systematically with the scale of predictions (heteroscedasticity).

#### B. Statistical Test: Breusch–Pagan Lagrange Multiplier Test
- **Auxiliary Model**: Regresses squared scaled residuals on all regressors $X_1, \dots, X_p$.
- **Hypotheses**:
  - $H_0: \text{Var}(\epsilon_i | X) = \sigma^2$ (Homoscedasticity)
  - $H_1: \text{Var}(\epsilon_i | X) = f(X)$ (Heteroscedasticity)
- **Decision Rule**: Reject $H_0$ if $p < 0.05$.
- **Remediation**: Use Huber–White Heteroscedasticity-Consistent standard errors (HC3) or Weighted Least Squares (WLS).

---

### 3.4 Category 4: Normality of Residuals

#### A. Graphical Diagnostic: Normal Q-Q Plot
- **Method**: Plots empirical standardized residual quantiles against standard theoretical normal quantiles $\Phi^{-1}(p_i)$.
- **Interpretation**: Adherence to the diagonal $45^\circ$ line indicates Gaussian behavior. Heavy S-curves indicate skewness; tail divergence indicates excess kurtosis.

#### B. Statistical Test: Jarque–Bera Test ($JB$)
$$JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right) \sim \chi^2(2)$$
where $S$ is sample skewness and $K$ is sample kurtosis.
- **Hypotheses**:
  - $H_0: S = 0 \text{ and } K = 3$ (Residual distribution has normal skewness and kurtosis)
  - $H_1$: Significant departure from normality
- **Decision Rule**: Reject $H_0$ if $p < 0.05$.

#### C. Statistical Test: Shapiro–Wilk Test
- Recommended for small-to-moderate sample sizes ($n < 2,000$).

---

### 3.5 Category 5: Absence of Multicollinearity

#### A. Metric: Variance Inflation Factor (VIF)
$$VIF_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination obtained from regressing feature $X_j$ on all other $p - 1$ predictors.
- **Standard Thresholds**:
  - $VIF = 1$: Perfect orthogonality.
  - $1 < VIF < 5$: Acceptable, benign correlation.
  - $VIF \ge 5$: High multicollinearity; standard error inflated by $\sqrt{VIF} \ge 2.23$.
  - $VIF \ge 10$: Severe multicollinearity; inversion instability and erratic coefficient signs.

#### B. Metric: Design Matrix Condition Number ($\kappa$)
$$\kappa(X) = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}$$
- **Threshold**: $\kappa > 30$ signals moderate to severe collinear conditioning.

---

## 4. Outliers, High Leverage & Influential Observations

1. **Hat Matrix Leverage ($h_{ii}$)**:
   - Measures extreme position in feature space $X$.
   - **Threshold**: Observations with $h_{ii} > \frac{2(p+1)}{n}$ have high leverage.
2. **Cook's Distance ($D_i$)**:
   - Quantifies the global shift in parameter estimates vector $\hat{\beta}$ when observation $i$ is omitted:
     $$D_i = \frac{e_i^2}{(p + 1) s^2} \left( \frac{h_{ii}}{(1 - h_{ii})^2} \right)$$
   - **Threshold**: Observations with $D_i > \frac{4}{n}$ (or $D_i > 1$) warrant careful qualitative inspection as influential data points.
