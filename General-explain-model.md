Here is the complete Chinese-English dual-language documentation and line-by-line code explanation for the predictive machine learning engine file `models.py`.

---

## Part 1: Chinese/English Dual-Language Process Report

The `StatisticalAdaptiveEngine` class acts as the **Predictive Machine Learning & Risk Boundary Calculator (预测性机器学习与风险边界计算器)** of the quantitative pipeline. Its primary responsibility is to handle multi-asset regression forecasting while explicitly mapping out non-parametric financial volatility. Instead of producing risky, single-point return estimates, it implements **Conformal Quantile Regression (CQR)**. This framework uses LightGBM models to construct adaptive, distribution-free asset return confidence intervals that adjust to changing market conditions.

以下是预测引擎与校准模型生命周期的详细解构：
Here is the step-by-step systemic machine learning and risk validation workflow:

### Step 1: Quantile Split Optimization Tuning / 初始化分位数回归弱估计器

* **What this step does / 此步骤的作用**: Sets up three separate LightGBM decision tree gradient boosting models for each financial symbol. Instead of minimizing regular squared errors, two of these models use a specialized pinball loss function to map upper and lower distribution boundaries.
* **Metrics & Calculations / 使用指标与计算**: Calculates precise target quantile coverage levels using the significance variable `ALPHA` (set to 0.05):

$$\text{Lower Quantile } (\alpha_{\text{low}}) = \frac{\alpha}{2} = \frac{0.05}{2} = 0.025 \quad (2.5\% \text{ Percentile}) \text{}$$


$$\text{Upper Quantile } (\alpha_{\text{high}}) = 1.0 - \frac{\alpha}{2} = 1.0 - 0.025 = 0.975 \quad (97.5\% \text{ Percentile}) \text{}$$


* **What data is used for / 数据用途**: Initializes the multi-model architecture. This allows the system to capture conditional heteroskedasticity (changing volatility trends) across individual assets.
* **What it returns / 返回值**: Three un-fitted `lgb.LGBMRegressor` model instances configured for lower quantiles, upper quantiles, and standard conditional means.

### Step 2: CQR Residual Calibration Matrix Compilation / 计算符合性非参数误差校准分数

* **What this step does / 此步骤的作用**: Fits the LightGBM models using rolling historical windows and computes non-parametric error metrics ($E_i$). These metrics quantify how much the model's initial interval boundaries miss the true, historical market returns.
* **Metrics & Calculations / 使用指标与计算**: Measures the maximum distance between predicted bounds and true values:

$$E_i = \max\left(\hat{q}_{\text{low}}(X_i) - y_i, \, y_i - \hat{q}_{\text{high}}(X_i)\right) \text{}$$



It then evaluates a high-order quantile across these compiled errors to calculate a protective buffer threshold:

$$\text{Q Error Threshold} = \text{Quantile}\left(\{E_i\}_{i=1}^T, \, 1.0 - \alpha\right) = \text{95th Percentile of Errors} \text{}$$


* **What data is used for / 数据用途**: Calculates statistical calibration adjustments. This allows the model to guarantee a $1 - \alpha$ (95%) probability that future asset returns will fall within the predicted intervals, regardless of market volatility distributions.
* **What it returns / 返回值**: Updates internal dictionary records (`self.models` and `self.cqr_thresholds`) with calibrated parameters for all tracked tokens.

### Step 3: Heteroskedastic Predictive Guard Slicing / 提取带边界的自适应置信极值

* **What this step does / 此步骤的作用**: Consumes the live, newest cross-sectional feature vectors to generate directional point returns along with precise conformal floors and ceilings for the upcoming trading day.
* **Metrics & Calculations / 使用指标与计算**: Extracts raw forecasts from the decision tree models and shifts the boundaries outward using the calibrated error threshold:

$$\text{Conformal Floor} = \text{Raw Prediction}_{\text{low}} - \text{Q Error Threshold} \text{}$$


$$\text{Conformal Ceiling} = \text{Raw Prediction}_{\text{high}} + \text{Q Error Threshold} \text{}$$


* **What data is used for / 数据用途**: Generates the primary prediction inputs for the downstream portfolio optimization module. The width of this interval gives the optimizer an asset-by-asset risk measurement.
* **What it returns / 返回值**: A structured data lookup dictionary (`predictions`) mapping ticker strings to tuples of floats: `(Point Mean, Conformal Floor, Conformal Ceiling)`.
