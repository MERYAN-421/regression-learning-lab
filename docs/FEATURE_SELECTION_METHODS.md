# Feature Selection Laboratory Methodology & Algorithms

This document provides technical, mathematical, and algorithmic specifications for the eight feature selection approaches implemented in **Milestone 2** of **Regression Learning Lab**.

---

## 1. Core Architectural Concepts

### 1.1 Original vs. Encoded Features
A fundamental rule in regression modeling is distinguishing between raw dataset predictors and preprocessed model matrix columns:
* **Original Features ($P$)**: The interpretable candidate predictor variables defined in the dataset catalog (e.g., `32` in Student Performance, `27` in Appliances Energy).
* **Encoded Dimensions ($D$)**: The expanded input matrix columns resulting from One-Hot Encoding categorical variables (e.g., Student Performance expands from `32` original features into `58` encoded columns).

> **Architectural Contract**: The user-facing parameter $K$ **strictly represents the count of selected original features**. Algorithms operate on or aggregate back to original feature groups, ensuring a categorical variable is never fragmented into disjoint dummy subsets.

---

## 2. Implemented Feature Selection Algorithms

| ID | Name | Category | Scope / Evaluation | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| `baseline` | All Features | Baseline | Uses all original predictors | $O(1)$ |
| `pearson` | Pearson Correlation | Filter | Target correlation magnitude | $O(N \cdot P)$ |
| `f_test` | F-test (`f_regression`) | Filter | Univariate linear regression F-statistic | $O(N \cdot P)$ |
| `mutual_information` | Mutual Information | Filter | Non-parametric entropy estimation | $O(N \cdot P \cdot \log N)$ |
| `rfe` | Recursive Feature Elimination | Wrapper | Iterative group-elimination via OLS | $O((P-K) \cdot N \cdot D^2)$ |
| `forward_sfs` | Forward Sequential Selection | Wrapper | Greedy greedy addition via CV RMSE | $O(K \cdot P \cdot N \cdot D_{\text{sub}}^2)$ |
| `backward_sfs` | Backward Sequential Selection | Wrapper | Greedy elimination via CV RMSE | $O((P-K) \cdot P \cdot N \cdot D_{\text{sub}}^2)$ |
| `lasso` | Lasso L1 Regularization | Embedded | L1 shrinkage with group L2 norm ranking | $O(N \cdot D \cdot \text{iterations})$ |

---

## 3. Method Specifications & Mathematical Formulations

### 3.1 Baseline — All Features (`baseline`)
* **Definition**: Retains all $P$ original predictors without screening or elimination.
* **Role**: Serves as the control reference benchmark to determine whether subset selection mitigates overfitting or improves validation generalization.

### 3.2 Pearson Correlation (`pearson`)
* **Univariate Filter**: Evaluates individual linear association with the continuous target:
  $$r_j = \frac{\sum_{i=1}^N (x_{ij} - \bar{x}_j)(y_i - \bar{y})}{\sqrt{\sum_{i=1}^N (x_{ij} - \bar{x}_j)^2} \sqrt{\sum_{i=1}^N (y_i - \bar{y})^2}}$$
* **Categorical Group Aggregation**: For a categorical feature $g$ spanning dummy columns $\{d_1, \dots, d_m\}$, the group score is the maximum absolute correlation among its levels:
  $$S_g = \max_{j \in \{d_1, \dots, d_m\}} |r_j|$$
* **Ranking**: Features are sorted by $S_g$ descending, selecting top-$K$.

### 3.3 F-Test (`f_test`)
* **Univariate Linear Screening**: Calculates the univariate regression F-statistic via scikit-learn `f_regression`:
  $$F_j = \frac{r_j^2 / 1}{(1 - r_j^2) / (N - 2)}$$
* **Group Score**: Max F-statistic across dummy columns:
  $$F_g = \max_{j \in g} F_j$$
* **Distinction**: Univariate screening $F$-test evaluates one predictor at a time against $y$; it must not be confused with the full-model OLS overall $F$-test.

### 3.4 Mutual Information (`mutual_information`)
* **Non-Linear Dependency Filter**: Measures information shared between predictor $X$ and target $Y$ via k-nearest neighbors entropy estimation (Kraskov-Stögbauer-Grassberger):
  $$I(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$
* **Group Score**: Max MI score among one-hot dummies:
  $$I_g = \max_{j \in g} I_j$$
* **Advantage**: Detects non-linear dependencies missed by Pearson correlation. Fixed random seed ensures reproducible entropy estimates.

### 3.5 Group-Aware Recursive Feature Elimination (`rfe`)
* **Wrapper Algorithm**:
  1. Initialize active pool with all $P$ original predictor groups.
  2. Fit `LinearRegression` on the encoded representation of the active pool.
  3. Compute each group's importance using the Euclidean $L_2$ norm of its regression coefficients:
     $$\|\boldsymbol{\beta}_g\|_2 = \sqrt{\sum_{j \in g} \beta_j^2}$$
  4. Eliminate the original feature group with the smallest $\|\boldsymbol{\beta}_g\|_2$.
  5. Repeat until exactly $K$ original groups remain.
* **Key Guarantee**: Dummy columns belonging to a categorical variable are eliminated as a single unit.

### 3.6 Forward Sequential Feature Selection (`forward_sfs`)
* **Wrapper Algorithm**:
  1. Start with an empty selected set $S = \emptyset$.
  2. For each candidate group $g \notin S$:
     * Fit `LinearRegression` on $S \cup \{g\}$ across $M$-fold cross-validation.
     * Compute average CV RMSE.
  3. Greedily select the group yielding the lowest CV RMSE:
     $$g^* = \arg\min_{g \notin S} \text{CV-RMSE}(S \cup \{g\})$$
  4. Add $g^*$ to $S$.
  5. Repeat until $|S| = K$.

### 3.7 Backward Sequential Feature Selection (`backward_sfs`)
* **Wrapper Algorithm**:
  1. Start with all original features $S = \{1, 2, \dots, P\}$.
  2. At each step, evaluate removing each candidate group $g \in S$.
  3. Greedily eliminate the group whose removal produces the best (lowest) CV RMSE:
     $$g^* = \arg\min_{g \in S} \text{CV-RMSE}(S \setminus \{g\})$$
  4. Repeat until $|S| = K$.

### 3.8 Lasso-Based Selection (`lasso`)
* **Embedded Regularization**: Minimizes the $L_1$-penalized least squares objective:
  $$\min_{\boldsymbol{\beta}} \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \sum_{j=1}^D |\beta_j|$$
* **Group Evaluation**: Group importance is evaluated via group coefficient magnitude:
  $$S_g = \sqrt{\sum_{j \in g} \hat{\beta}_j^2}$$
* **Natural Sparsity vs. Exact-K Limitation**:
  > **Pedagogical Note**: Ordinary Lasso induces coefficient-level sparsity naturally via $\alpha$; it does not natively enforce an exact group count $K$. The lab estimates optimal $\alpha$ via internal 3-fold cross-validation, records the natural non-zero group count, and provides top-$K$ group selection as a documented post-hoc ranking. The UI explicitly alerts users to this distinction.
