
## Part 1: Chinese/English Dual-Language Process Report

The `DataProcessor` class serves as the **Feature Engineering & Data Cleansing Engine (特征工程与数据清洗引擎)** of the quant pipeline. Its primary role is to ingest raw multi-asset OHLCV (Open, High, Low, Close, Volume) market data, neutralize long-term inflation artifacts, compute mathematically sound risk/momentum signals, and assemble structured cross-sectional panels ready for statistical modeling.

以下是系统数据处理生命周期的详细解构：
Here is the step-by-step systemic validation and calculation workflow:

### Step 1: Real-Value Purchasing Power Adjustment / 真实购买力通胀平减

* **What this step does / 此步骤的作用**: Modifies nominal market asset price history by accounting for currency depreciation over time. It discounts historical quotes against an annualized constant rate relative to the maximum observed timestamp in the dataset.
* **Metrics & Calculations / 使用指标与计算**: Calculates chronological distance fractions in years, converting nominal variables into stable real variants:

$$\text{Years Back} = \frac{\text{Max Date} - \text{Current Date}}{365.25}$$


$$\text{Deflator} = (1.0 + \text{ANNUAL\_INFLATION\_RATE})^{\text{Years Back}}$$


$$\text{Real Price} = \text{Nominal Price} \times \text{Deflator}$$


* **What data is used for / 数据用途**: Suppresses structural macroeconomic non-stationarity drift and inflationary variance distortion, aligning raw numerical properties over multi-year scales.
* **What it returns / 返回值**: A cloned pandas DataFrame containing newly appended deflation-adjusted columns (`Real_Open`, `Real_High`, `Real_Low`, `Real_Close`).

### Step 2: Forward Target Return Generation / 构建前瞻对数真实目标收益率

* **What this step does / 此步骤的作用**: Formulates an asset-specific prediction target column ($y$) using next-day forward log returns. This provides the ground truth training labels for the downstream Conformal Quantile Regression (CQR) machine learning model.
* **Metrics & Calculations / 使用指标与计算**: Implements protective divisor limits to guarantee numerical stability before evaluating a shifted log vector:

$$\text{Safe Prev Close} = \max(\text{Real Close}_{t-1}, 10^{-8})$$


$$\text{Target Return}_t = \ln\left(\frac{\text{Real Close}_t}{\text{Safe Prev Close}}\right)_{t+1}$$


* **What data is used for / 数据用途**: Maps localized market setups directly to their subsequent multi-asset directional outcomes, forming the baseline label tracking array.
* **What it returns / 返回值**: A computed column (`target_return`) assigned to an internal temporary processing dataframe.

### Step 3: White-Box Explanatory Feature Engineering / 显式白盒特征提取

* **What this step does / 此步骤的作用**: Computes the five explicit indicators declared within the global configuration settings to build a mathematical profile of momentum, volatility, and volume dynamics.
* **Metrics & Calculations / 使用指标与计算**:
* **Multi-period Momentum (`Mom_1D`, `Mom_5D`, `Mom_20D`)**: $\ln(\text{Real Close}_t / \text{Real Close}_{t-k})$ for $k \in \{1, 5, 20\}$.
* **Garman-Klass Volatility (`GK_Vol`)**: Incorporates open, high, low, and close levels into a specialized non-parametric tracking variance signature:

$$\text{GK\_Vol} = 0.5 \times \left(\ln\frac{\text{High}}{\text{Low}}\right)^2 - (2\ln 2 - 1) \times \left(\ln\frac{\text{Close}}{\text{Open}}\right)^2$$


* **Volume Turnover Shock (`Turnover_Shock`)**: Normalized tracking volume signals evaluated against a 5-day moving average:

$$\text{Turnover\_Shock} = \frac{\text{Turnover}_t}{\text{Rolling Mean}(\text{Turnover}, 5)}$$




* **What data is used for / 数据用途**: Translates basic OHLCV structural price lines into a multi-factor state representation matrix ($X$) optimized for machine learning classifiers.
* **What it returns / 返回值**: Updated multi-column metric inputs inside a temporary processing dataframe.

### Step 4: Strict Filtering & Multi-Indexed Panel Consolidation / 样本安全清洗与多级索引面板合并

* **What this step does / 此步骤的作用**: Performs rigorous data cleansing on feature rows. It filters out missing values (NaNs), strips away non-finite computing anomalies, evaluates sample size safety bounds, and merges independent asset frames into unified cross-sectional matrices.
* **Metrics & Calculations / 使用指标与计算**: Asserts validation filters across the feature collection space, dropping rows if:

$$\text{Valid Sample Count} < \text{MIN\_REQUIRED\_SAMPLES}$$


* **What data is used for / 数据用途**: Ensures structural data integrity. This safeguards down-line training loops and optimization layers against matrix singularization issues caused by missing values or insufficient samples.
* **What it returns / 返回值**: A dictionary tuple consisting of the Multi-Indexed Feature Space Panel Matrix (`X_panel`) and the matched Ground Truth Label Vector (`y_panel`), structured by continuous coordinate index dimensions `[Date, Symbol]`.

---
